# src/capture/capture_live.py
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
import json
from datetime import datetime

# --- CONFIG ---
OUT_DIR = os.path.join(os.getcwd(), "data")
LANDMARKS_DIR = os.path.join(OUT_DIR, "landmarks")
IMAGES_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(LANDMARKS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

CSV_PATH = os.path.join(LANDMARKS_DIR, "landmarks.csv")
SAVE_IMAGES = True  # set False if you don't want to save full images

# key -> label mapping (change as needed)
LABELS = {
    ord('1'): "Anjali",   # press '1' to label Anjali mudra
    ord('2'): "Gyan",
    ord('3'): "Prithvi",
    ord('4'): "Varun",
    ord('5'): "Surya",
    ord('0'): "None"      # press '0' for no label / capture as background
}

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize CSV (header)
if not os.path.exists(CSV_PATH):
    # build header: frame_id,label,handedness,timestamp,user_id,x0,y0,z0,...x20,y20,z20,notes
    coords = []
    for i in range(21):
        coords += [f"x{i}", f"y{i}", f"z{i}"]
    header = ["frame_id", "label", "handedness", "timestamp", "user_id"] + coords + ["image_path", "notes"]
    pd.DataFrame(columns=header).to_csv(CSV_PATH, index=False)

def landmarks_to_list(landmarks):
    # landmarks: list of mp normalized points
    out = []
    for p in landmarks:
        out.extend([p.x, p.y, p.z])
    return out

def main(user_id="user01"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        print("Press keys 1..5 to label mudras, 0=none, q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            display = frame.copy()
            label_to_save = None
            handedness = "NA"
            lm_list = [np.nan] * (21*3*2)

            if results.multi_hand_landmarks:
                handedness_list = []
                for idx, (hand_landmarks, hand_handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    mp_drawing.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    handedness_list.append(hand_handedness.classification[0].label)
                    # Fill landmarks for this hand
                    hand_coords = landmarks_to_list(hand_landmarks.landmark)  # 21*3
                    start = idx*21*3
                    lm_list[start:start+21*3] = hand_coords
                handedness = ",".join(handedness_list)

            cv2.putText(display, f"User: {user_id}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(display, "Keys: " + " ".join([f"{chr(k)}:{v}" for k,v in LABELS.items() if k!=ord('0')]),
                        (10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.imshow("Mudra Capture", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key in LABELS:
                chosen_label = LABELS[key]
                ts = datetime.utcnow().isoformat()
                frame_id = f"{user_id}_{int(time.time()*1000)}"
                image_path = ""
                if SAVE_IMAGES:
                    image_filename = f"{frame_id}.jpg"
                    image_path = os.path.join("data", "images", image_filename)
                    cv2.imwrite(os.path.join(IMAGES_DIR, image_filename), frame)

                row = {
                    "frame_id": frame_id,
                    "label": chosen_label,
                    "handedness": handedness,
                    "timestamp": ts,
                    "user_id": user_id,
                    "image_path": image_path,
                    "notes": ""
                }
                # fill coordinates (21*3)
                if lm_list[0] is None:
                    # no hand detected; fill NaNs
                    coords = [np.nan]*(21*3)
                else:
                    coords = lm_list
                for i, v in enumerate(coords):
                    row[f"x{i}"] = v if i < len(coords) else np.nan

                # append to CSV
                df = pd.DataFrame([row])
                df.to_csv(CSV_PATH, mode='a', header=False, index=False)
                print(f"Saved frame {frame_id} label={chosen_label} hand={handedness}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # pass optional user id as arg
    import sys
    uid = sys.argv[1] if len(sys.argv) > 1 else "user01"
    main(user_id=uid)
