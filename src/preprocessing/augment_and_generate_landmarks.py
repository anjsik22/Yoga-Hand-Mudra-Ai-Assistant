import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from datetime import datetime
from tqdm import tqdm

# -------- CONFIG --------
BASE_DIR = os.getcwd()
DATA_DIRS = [
    os.path.join(BASE_DIR, "data", "images"),
    os.path.join(BASE_DIR, "data", "images_augmented")
]
LANDMARKS_DIR = os.path.join(BASE_DIR, "data", "landmarks")
os.makedirs(LANDMARKS_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(LANDMARKS_DIR, "landmarks_clean.csv")

# -------- MEDIAPIPE --------
mp_hands = mp.solutions.hands

def preprocess_image(img):
    """Resize, normalize, and enhance contrast."""
    img = cv2.resize(img, (320, 320))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # brightness correction
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

def get_landmarks(image_path):
    """Return 42*3 landmarks or np.nan list."""
    img = cv2.imread(image_path)
    if img is None:
        return [np.nan] * (42 * 3)

    img = preprocess_image(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.45) as hands:
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            # Try flipping (sometimes hand orientation confuses Mediapipe)
            img_flipped = cv2.flip(img_rgb, 1)
            results = hands.process(img_flipped)

        landmarks = [np.nan] * (42 * 3)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                start = idx * 21 * 3
                for i, lm in enumerate(hand_landmarks.landmark):
                    landmarks[start + i * 3] = lm.x
                    landmarks[start + i * 3 + 1] = lm.y
                    landmarks[start + i * 3 + 2] = lm.z
        return landmarks

# -------- MAIN --------
rows = []
for data_dir in DATA_DIRS:
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        print(f"\n📂 Processing class: {label} from {os.path.basename(data_dir)}")

        for img_file in tqdm(os.listdir(class_dir)):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_dir, img_file)
            lm_list = get_landmarks(img_path)

            row = {
                "frame_id": os.path.splitext(img_file)[0],
                "label": label,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": os.path.relpath(img_path, BASE_DIR)
            }

            for i in range(42):
                for axis, j in zip(['x', 'y', 'z'], range(3)):
                    row[f"{axis}{i}"] = lm_list[i * 3 + j]
            rows.append(row)

# -------- SAVE --------
df = pd.DataFrame(rows)

# filter only those with at least 1 valid landmark
cols = [c for c in df.columns if c.startswith('x')]
valid_rows = df[df[cols].notna().sum(axis=1) > 10]
valid_rows.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Cleaned landmarks saved: {OUTPUT_CSV}")
print(f"Total images processed: {len(df)}")
print(f"Valid hand detections: {len(valid_rows)} ({len(valid_rows)/len(df)*100:.2f}%)")
