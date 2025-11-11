# src/preprocessing/generate_landmarks_csv.py

import os
import pandas as pd
import numpy as np
from datetime import datetime

# --- Paths ---
BASE_DIR = os.getcwd()
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")  # contains subfolders per class
CSV_PATH = os.path.join(BASE_DIR, "data", "landmarks", "landmarks.csv")

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# --- Generate rows ---
rows = []

for label in os.listdir(IMAGES_DIR):
    class_dir = os.path.join(IMAGES_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            row = {
                "frame_id": os.path.splitext(img_file)[0],
                "label": label,
                "handedness": np.nan,     # optional, fill later if available
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": "unknown",
                "image_path": os.path.join("data", "images", label, img_file)
            }
            # Add placeholder for 42 landmarks (x0..z20 for 2 hands)
            for i in range(42):
                for axis in ['x', 'y', 'z']:
                    row[f"{axis}{i}"] = np.nan
            rows.append(row)

# --- Create DataFrame ---
df = pd.DataFrame(rows)

# --- Save CSV ---
df.to_csv(CSV_PATH, index=False)
print(f"✅ landmarks.csv created with {len(df)} rows at: {CSV_PATH}")
