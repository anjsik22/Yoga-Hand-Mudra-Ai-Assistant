# src/preprocessing/normalize.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import joblib

# ---------------- CONFIG ----------------
BASE_DIR = os.getcwd()
LANDMARKS_CSV = os.path.join(BASE_DIR, "data/landmarks/landmarks_clean.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/preprocessed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD CSV ----------------
df = pd.read_csv(LANDMARKS_CSV)
print(f"✅ Loaded {len(df)} rows from {LANDMARKS_CSV}")

# ---------------- SELECT LANDMARK COLUMNS ----------------
landmark_cols = [c for c in df.columns if c.startswith(('x','y','z'))]
X_df = df[landmark_cols]

# ---------------- FILTER ROWS WHERE BOTH HANDS MISSING ----------------
# Hand1 = first 21 points, Hand2 = next 21 points
hand1_cols = landmark_cols[:63]
hand2_cols = landmark_cols[63:]

valid_rows_mask = ~(X_df[hand1_cols].isna().all(axis=1) & X_df[hand2_cols].isna().all(axis=1))
df = df[valid_rows_mask].reset_index(drop=True)
X_df = X_df.loc[valid_rows_mask]
print(f"📍 {len(df)} rows with at least one hand detected will be used")

# ---------------- FILL PARTIAL MISSING VALUES ----------------
def fill_hand_nan(hand_array):
    """Fill NaNs for a hand with column mean"""
    nan_mask = np.isnan(hand_array)
    col_means = np.nanmean(hand_array, axis=0)
    # Replace NaNs with column means
    hand_array[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return hand_array

X_array = X_df.values
# Process Hand1 and Hand2 separately
for i in range(X_array.shape[0]):
    # Hand1
    hand1 = X_array[i,:63].reshape(21,3)
    if np.isnan(hand1).all():
        hand1[:] = 0  # completely missing hand → zeros
    else:
        hand1 = fill_hand_nan(hand1)
    X_array[i,:63] = hand1.flatten()
    
    # Hand2
    hand2 = X_array[i,63:].reshape(21,3)
    if np.isnan(hand2).all():
        hand2[:] = 0
    else:
        hand2 = fill_hand_nan(hand2)
    X_array[i,63:] = hand2.flatten()

# ---------------- NORMALIZE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array)
print("📊 Landmarks normalized with StandardScaler")

# Save the scaler
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.save")
joblib.dump(scaler, SCALER_PATH)
print(f"💾 Scaler saved to: {SCALER_PATH}")

# ---------------- ENCODE LABELS ----------------
if 'label' not in df.columns:
    raise KeyError("❌ 'label' column missing in CSV.")

le = LabelEncoder()
y = le.fit_transform(df['label'].astype(str))
print(f"🏷️  Labels encoded: {list(le.classes_)}")

# ---------------- SAVE OUTPUTS ----------------
np.save(os.path.join(OUTPUT_DIR, "X.npy"), X_scaled)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

label_map = {int(i): label for i, label in enumerate(le.classes_)}
with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w") as f:
    json.dump(label_map, f, indent=2)

print(f"\n✅ Preprocessing complete!")
print(f"📦 X shape: {X_scaled.shape}, y shape: {y.shape}")
print(f"💾 Preprocessed files saved to: {OUTPUT_DIR}")
