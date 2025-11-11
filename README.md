# 🧘 AI-Based Yoga Mudra Detection System

This project detects, analyzes, and corrects **Yoga Hand Mudras** in real time using AI and computer vision.

## 🧠 Features
- Real-time hand tracking with MediaPipe
- Mudra classification with Deep Learning
- Feedback system for correction
- Supports multiple users and poses

## ⚙️ Technologies
- Python 3.10+
- OpenCV, MediaPipe, TensorFlow, NumPy, Pandas
- VS Code for development

## 📁 Project Structure

```text
YOGA-MUDRA/
│
├── checkpoints/                     # Trained model weights
│   ├── feedback_model.pth
│   └── mudra_model_best.pth
│
├── data/                            # Dataset and processed files
│   ├── images/                      # Captured + downloaded yoga hand mudra images
│   ├── images_augmented/            # Augmented (AI-generated) images
│   ├── landmarks/                   # Landmark data extracted using MediaPipe
│   │   └── landmarks_clean.csv
│   └── preprocessed/                # Normalized and encoded training data
│       ├── X.npy
│       ├── y.npy
│       ├── label_mapping.json
│       └── scaler.save
│
├── src/                             # Source code modules
│   ├── capture/                     # Capture and prediction scripts
│   │   ├── capture_live.py          # Capture and label mudras manually
│   │   ├── capture_live_predict.py  # Real-time mudra prediction (model inference)
│   │   └── capture_live_assistant.py# AI-based mudra assistant with feedback (audio + visual)
│   │
│   ├── models/                      # Model training and architecture scripts
│   │   ├── train.py                 # Train mudra classification model
│   │   └── train_feedback_models.py # Train feedback correction model
│   │
│   ├── preprocessing/               # Data preparation and cleaning scripts
│   │   ├── augment_and_generate_landmarks.py  # Data augmentation + landmark generation
│   │   ├── clean_landmarks.py                # Clean raw landmark CSVs
│   │   ├── generate_landmarks_csv.py         # Generate landmark dataset from images
│   │   └── normalize.py                      # Normalize and encode dataset for training
│   │
│   └── utils/                       # (Optional) helper utilities
│
├── venv/                            # Virtual environment (ignored in Git)
│
├── .gitignore                       # Ignored files/folders list
├── README.md                        # Project documentation
└── requirements.txt                 # Dependencies list


▶️ How to Run

1. Activate your virtual environment

```bash .\venv\Scripts\activate


2. Run the live capture script to start collecting mudra data:

```bash python src/capture/capture_live.py user01


3. Run real-time AI assistant (for detection and feedback):

```bash python src/capture/capture_live_assistant.py

🚀 Next Steps

 Collect additional dataset using MediaPipe

 Train and fine-tune MudraNet model for higher accuracy

 Integrate advanced AI-driven correction feedback

 Explore web or mobile deployment for wider usability

🧩 Credits

Developed as part of an AI-driven Yoga Hand Mudra Assistant Project,
combining computer vision, deep learning, and wellness innovation.

✅ Why this version is better:

✔️ Clean headings and emoji icons
✔️ Proper code block indentation
✔️ Works beautifully on GitHub’s markdown renderer
✔️ Readable section spacing and professional layout