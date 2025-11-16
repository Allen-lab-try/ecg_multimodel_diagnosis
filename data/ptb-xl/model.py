# ============================================================
# model_loader.py  —  Run ECGFounder to extract 1024 features
# ============================================================
import torch
import numpy as np
from ecg_loader import load_ptbxl_subset
from finetune_model import ft_12lead_ECGFounder
from tqdm import tqdm

# ------------------------------------------------------------
# 1️⃣ Parameter Settings
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pth = r"E:\professor_yen\ECGFounder\checkpoint\12_lead_ECGFounder.pth"
path = r"E:\professor_yen\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

num_records = 500        # Load a subset first for testing
max_workers = 8
batch_size = 8

# ------------------------------------------------------------
# 2️⃣ Load PTB-XL data
# ------------------------------------------------------------
print("🚀 Loading PTB-XL subset...")
X, Y = load_ptbxl_subset(path, num_records=num_records, max_workers=max_workers)
print(f"✅ Loaded: X={X.shape}, Y={len(Y)}")

# ------------------------------------------------------------
# 3️⃣ Load ECGFounder model
# ------------------------------------------------------------
print("🚀 Loading ECGFounder model from:", pth)
model = ft_12lead_ECGFounder(device, pth, n_classes=150, linear_prob=False)
model.eval()
model.return_features = True   # Enable output of 1024-d feature vectors

# ------------------------------------------------------------
# 4️⃣ Extract 1024-dimensional Features
# ------------------------------------------------------------
feats_list, logits_list = [], []
with torch.no_grad():
    for i in tqdm(range(0, len(X), batch_size), desc="Extracting features"):
        batch = X[i:i+batch_size]             # shape (B, 5000, 12)
        batch = np.transpose(batch, (0, 2, 1))  # change axis to (B, 12, 5000)
        batch = torch.tensor(batch, dtype=torch.float32).to(device)

        logits, feats = model(batch)
        feats_list.append(feats.cpu().numpy())
        logits_list.append(logits.cpu().numpy())

# Concatenate into full matrices
features_1024 = np.concatenate(feats_list, axis=0)
logits_all = np.concatenate(logits_list, axis=0)
print(f"✅ Features shape: {features_1024.shape}")
print(f"✅ Logits shape: {logits_all.shape}")

# ------------------------------------------------------------
# 5️⃣ Save extracted features for later analysis
# ------------------------------------------------------------
np.save("features_1024.npy", features_1024)
np.save("logits.npy", logits_all)
Y.to_csv("labels_meta.csv")

print("💾 Saved features_1024.npy, logits.npy, labels_meta.csv")


