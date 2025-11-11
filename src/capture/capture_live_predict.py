# src/capture/capture_live_predict.py
import cv2
import mediapipe as mp
import numpy as np
import torch
import os
import json
import joblib

# --- CONFIG ---
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "mudra_model_best.pth")
SCALER_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "scaler.save")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "label_mapping.json")

# --- LOAD LABEL MAPPING ---
with open(LABEL_MAP_PATH, "r") as f:
    label_mapping = json.load(f)

# --- LOAD SCALER ---
scaler = joblib.load(SCALER_PATH)

# --- MODEL DEFINITION ---
import torch.nn as nn

class MudraNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

# --- LOAD MODEL ---
input_dim = 21*3*2  # 2 hands, 21 landmarks, 3 coords
num_classes = len(label_mapping)
model = MudraNet(input_dim, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
    """Extract 2-hand landmarks as 126-length list"""
    landmarks = [0] * (21*3*2)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= 2:  # max 2 hands
                break
            base = idx * 21 * 3
            for i, lm in enumerate(hand_landmarks.landmark):
                landmarks[base + i*3] = lm.x
                landmarks[base + i*3 + 1] = lm.y
                landmarks[base + i*3 + 2] = lm.z
    return landmarks

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks & predict
            landmarks = extract_landmarks(results)
            landmarks_scaled = scaler.transform([landmarks])
            input_tensor = torch.tensor(landmarks_scaled, dtype=torch.float32)

            with torch.no_grad():
                output = model(input_tensor)
                pred_class = torch.argmax(output, dim=1).item()
                pred_label = label_mapping[str(pred_class)]

            # Display
            cv2.putText(frame, f"Prediction: {pred_label}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Mudra Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
