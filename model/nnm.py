import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Auto-detect CUDA GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing PyTorch Engine on Device: {device}")

# ==========================================
# 1. DEEP AUTOENCODER ARCHITECTURE
# ==========================================
class ProcurementAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(ProcurementAutoencoder, self).__init__()

        # Encoder: compresses features into latent representation.
        # Widened vs. the original 12/6/3 because we now feed it ~50 columns
        # (one-hot category/department + all engineered signals), not 4.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),  # bottleneck
        )

        # Decoder: reconstructs original feature space
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ==========================================
# 2. TRAINING & EVALUATION PIPELINE
# ==========================================
def run_phase_2_training(csv_path="procurement_training_dataset.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' not found. Run the dataset generator first!")

    df = pd.read_csv(csv_path)

    # ---- feature prep -----------------------------------------------------
    # The dataset's real columns (not the placeholder bid_ratio/submission_delay_min/
    # text_perplexity/shared_pan_count from the first draft). Drop IDs and the
    # label columns -- the label is used below ONLY to select the training subset
    # and to validate afterward, never as a model input.
    id_cols = ["bid_id", "tender_id", "vendor_id", "contract_id"]
    label_cols = ["fraud_category", "is_anomaly"]
    cat_cols = ["category", "department_id"]

    X_df = df.drop(columns=id_cols + label_cols)
    X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
    bool_cols = X_df.select_dtypes(include="bool").columns
    X_df[bool_cols] = X_df[bool_cols].astype(int)
    feature_cols = X_df.columns.tolist()

    X = X_df.values.astype(np.float32)
    is_anomaly = df["is_anomaly"].values
    normal_mask = is_anomaly == 0

    # ---- KEY FIX: fit scaler and train the autoencoder on NORMAL rows only.
    # An autoencoder's anomaly signal comes from it being *bad* at reconstructing
    # patterns it never saw. Training on the full mixed dataset (as in the
    # original script) teaches it to reconstruct fraud patterns too, which
    # flattens the reconstruction-error gap between normal and fraudulent rows.
    scaler = StandardScaler()
    scaler.fit(X[normal_mask])
    X_scaled = scaler.transform(X)

    tensor_all = torch.FloatTensor(X_scaled).to(device)
    tensor_train = torch.FloatTensor(X_scaled[normal_mask]).to(device)

    # Initialize model
    input_dim = X_scaled.shape[1]
    model = ProcurementAutoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train unsupervised autoencoder on normal-only baseline
    epochs = 200
    model.train()
    print(f"\nTraining Deep Autoencoder on {tensor_train.shape[0]} normal-baseline rows "
          f"({tensor_train.shape[0]}/{tensor_all.shape[0]} total)...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        reconstructed = model(tensor_train)
        loss = criterion(reconstructed, tensor_train)
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            print(f"Epoch [{epoch}/{epochs}] - Reconstruction Loss (MSE): {loss.item():.6f}")

    # ---- score every row (normal + fraud) using the trained model --------
    model.eval()
    with torch.no_grad():
        reconstructions = model(tensor_all)
        mse = torch.mean((tensor_all - reconstructions) ** 2, dim=1).cpu().numpy()

    # Normalize MSE to a 0-100 risk score
    risk_scores = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8) * 100
    df["Risk_Score"] = np.round(risk_scores, 2)

    # Threshold derived from the NORMAL distribution (95th percentile of normal
    # reconstruction error), not an arbitrary fixed constant -- this adapts if
    # you regenerate the dataset with a different scale/anomaly rate.
    normal_mse = mse[normal_mask]
    mse_threshold = np.percentile(normal_mse, 95)
    risk_threshold = (mse_threshold - mse.min()) / (mse.max() - mse.min() + 1e-8) * 100
    df["Neural_Flag"] = df["Risk_Score"] > risk_threshold

    # ---- validate against known labels (evaluation only, never used in training)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(is_anomaly, mse)
    flag_rate_normal = df.loc[normal_mask, "Neural_Flag"].mean()
    flag_rate_fraud = df.loc[~normal_mask, "Neural_Flag"].mean()

    # Save output for Dashboard UI
    output_path = "audit_ai_scored.csv"
    df.to_csv(output_path, index=False)

    # Save PyTorch model checkpoint + preprocessing so inference matches training
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "risk_threshold": float(risk_threshold),
    }, "autoencoder_model.pth")

    print("\n==========================================")
    print("PHASE 2 COMPLETE: PYTORCH MODEL TRAINED")
    print("==========================================")
    print(f"Scored dataset saved to: '{output_path}'")
    print(f"Model checkpoint saved to: 'autoencoder_model.pth'")
    print(f"Total Suspicious Flags Triggered: {df['Neural_Flag'].sum()} / {len(df)}")
    print(f"Reconstruction-error AUC (normal vs fraud): {auc:.3f}")
    print(f"Flag rate on normal rows: {flag_rate_normal:.3f} (false-alarm rate)")
    print(f"Flag rate on fraud rows:  {flag_rate_fraud:.3f} (catch rate)")

    return df


if __name__ == "__main__":
    run_phase_2_training()