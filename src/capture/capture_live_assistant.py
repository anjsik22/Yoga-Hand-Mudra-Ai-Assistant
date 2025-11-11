import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
import os
import joblib

# ---------------- CONFIG ----------------
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "mudra_model_best.pth")
SCALER_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "scaler.save")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "label_mapping.json")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------- LOAD MODEL ----------------
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

with open(LABEL_MAP_PATH, "r") as f:
    label_mapping = json.load(f)

num_classes = len(label_mapping)
input_dim = 21*3*2
model = MudraNet(input_dim, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

scaler = joblib.load(SCALER_PATH)

# ---------------- HELPERS ----------------
FINGER_MAP = {
    "Thumb": list(range(0, 5)),
    "Index": list(range(5, 9)),
    "Middle": list(range(9, 13)),
    "Ring": list(range(13, 17)),
    "Pinky": list(range(17, 21))
}

def extract_landmarks(results):
    lm_list = [0.0] * 126
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            start = idx*21*3
            for i, lm in enumerate(hand_landmarks.landmark):
                if start + i*3 + 2 < 126:
                    lm_list[start + i*3] = lm.x
                    lm_list[start + i*3 + 1] = lm.y
                    lm_list[start + i*3 + 2] = lm.z
    return lm_list

def generate_feedback(landmarks):
    """AI-style feedback: detect which fingers are misaligned or open"""
    feedback = []
    landmarks = np.array(landmarks).reshape(21,3)

    # Compute relative distances between tips and base joints
    for finger, indices in FINGER_MAP.items():
        tip = landmarks[indices[-1]]
        base = landmarks[indices[0]]
        dist = np.linalg.norm(tip - base)
        
        # Empirical thresholds for openness
        if dist > 0.35:
            feedback.append(f"Relax your {finger.lower()} finger")
        elif dist < 0.15:
            feedback.append(f"Straighten your {finger.lower()} finger")
    
    if not feedback:
        feedback.append("Excellent! Your mudra looks correct ✅")
    return feedback

# ---------------- MAIN ----------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

                lm_list = extract_landmarks(results)
                X_input = np.array(lm_list).reshape(1, -1)
                X_scaled = scaler.transform(X_input)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

                with torch.no_grad():
                    outputs = model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred_idx = torch.max(probs, 1)
                    confidence = confidence.item() * 100
                    mudra_name = label_mapping[str(pred_idx.item())]

                if confidence < 70:
                    feedback = generate_feedback(lm_list[:63])
                    cv2.putText(frame, f"Feedback ({confidence:.1f}%):", (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    for i, f in enumerate(feedback):
                        cv2.putText(frame, f, (10, 90 + 30*i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                else:
                    cv2.putText(frame, f"Mudra: {mudra_name}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    cv2.putText(frame, "Excellent! Your mudra looks correct ✅", (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            else:
                cv2.putText(frame, "No hand detected", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            cv2.imshow("Yoga Mudra AI Assistant", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
