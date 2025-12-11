📘 ACTIVITY RECOGNITION USING CNN, LSTM & PYTORCH

This project implements a video action recognition system using a hybrid CNN + LSTM architecture in PyTorch.

Each video frame is passed through a MobileNetV2 CNN to extract features.

A LSTM processes the temporal sequence of these features.

A classification head predicts the action class for each video clip.

📁 Repository Structure
.
├── model.py       # CNN + LSTM action recognition model
├── dataset.py     # Video frame dataset loader and preprocessing
├── train.py       # Training + validation pipeline
└── data/          # HMDB51-style dataset split (train/val)

🧠 Model Architecture
1️⃣ CNN Feature Extractor (MobileNetV2)

Pretrained on ImageNet (optional)

Extracts a 1280-dimensional feature vector per frame

Backbone can be frozen or fine-tuned

2️⃣ LSTM Sequence Model

Input shape: (batch, time_steps, feature_dim)

Adjustable:

Hidden size

Number of layers

Bidirectional or unidirectional

Learns temporal patterns across video frames

3️⃣ Classification Head

Fully-connected layers

Dropout for regularization

Outputs logits for num_classes actions

📦 Dataset Structure

Dataset must follow an HMDB51-like folder structure:

data/
└── HMDB51_split/
    ├── train/
    │   ├── jump/
    │   ├── walk/
    │   ├── run/
    │   └── sit/
    └── val/
        ├── jump/
        ├── walk/
        ├── run/
        └── sit/


Each class folder contains multiple videos, and each video folder contains extracted frames:

walk/
   video_01/
       frame_0001.jpg
       frame_0002.jpg
       ...
   video_02/
       frame_0001.jpg
       ...
