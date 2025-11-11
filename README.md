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


### How to Run
1. Activate your virtual environment  
2. Run the capture script to start collecting data:
   \`\`\`
   python src/capture/capture_live.py user01
   \`\`\`

### Next Steps
- [ ] Collect dataset using MediaPipe
- [ ] Train baseline mudra classifier
- [ ] Add real-time correction feedback
" > README.md