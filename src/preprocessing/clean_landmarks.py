# src/preprocessing/clean_landmarks.py

import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Paths ---
BASE_DIR = os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "data", "landmarks", "landmarks.csv")  # original captured images
CLEAN_CSV_PATH = os.path.join(BASE_DIR, "data", "landmarks", "landmarks_clean.csv")
AUG_IMAGES_DIR = os.path.join(BASE_DIR, "data", "images_augmented")  # augmented images per class

os.makedirs(os.path.dirname(CLEAN_CSV_PATH), exist_ok=True)

# --- Step 1: Load original landmarks CSV ---
try:
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
except Exception as e:
    print(f"❌ Error reading CSV: {e}")
    exit()

print(f"✅ Loaded {len(df)} rows from original landmarks.csv")

# --- Step 2: Add augmented images ---
aug_rows = []
if os.path.exists(AUG_IMAGES_DIR):
    for label in os.listdir(AUG_IMAGES_DIR):
        class_dir = os.path.join(AUG_IMAGES_DIR, label)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                row = {
                    "frame_id": os.path.splitext(img_file)[0],
                    "label": label,
                    "handedness": np.nan,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user_id": "augmented",
                    "image_path": os.path.join("data", "images_augmented", label, img_file),
                    "notes": "augmented"
                }
                # Placeholder for 42 landmarks (x0..z20 for 2 hands)
                for i in range(42):
                    for axis in ['x', 'y', 'z']:
                        row[f"{axis}{i}"] = np.nan
                aug_rows.append(row)
    print(f"✅ Added {len(aug_rows)} rows from augmented images")

# Append augmented rows to original dataframe
if aug_rows:
    df = pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True)

# --- Step 3: Ensure all landmark columns exist ---
landmark_cols = [f"{axis}{i}" for i in range(42) for axis in ['x', 'y', 'z']]
for col in landmark_cols:
    if col not in df.columns:
        df[col] = np.nan

# --- Step 4: Reorder columns ---
fixed_columns = ['frame_id', 'label', 'handedness', 'timestamp', 'user_id']
optional_columns = ['image_path', 'notes']
df = df[[col for col in fixed_columns + landmark_cols + optional_columns if col in df.columns]]

# --- Step 5: Save cleaned CSV ---
df.to_csv(CLEAN_CSV_PATH, index=False)
print(f"✅ Cleaned CSV saved to: {CLEAN_CSV_PATH}")
print(f"Total rows (captured + augmented): {len(df)}")
