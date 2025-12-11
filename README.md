## ACTIVITY RECOGNITION USING CNN, LSTM and PYTORCH

This project implements a video action recognition model using a CNN + LSTM hybrid architecture in PyTorch.
Each frame is passed through MobileNetV2 to extract per-frame features, and then an LSTM learns temporal patterns across frames.
The final classifier predicts the action class for each video.

# This repository contains:
1. model.py → CNN-LSTM action recognition model
2. dataset.py → Video frame loading + preprocessing
3. train.py → Training & validation pipeline
4. data/ → Folder containing HMDB51-style split dataset
--
# Model Architecture 

1. CNN Feature Extractor
Uses MobileNetV2 as a backbone
Pretrained weights (optional)
Outputs a 1280-dim feature vector per frame
Can freeze or finetune CNN layers

2. LSTM Sequence Model
Takes feature sequences shaped (batch, time, feature_dim)
Options:
LSTM hidden size
1 or more layers
Bidirectional support
Learns temporal dependencies across video frames

3. Classifier
Fully connected layers
Dropout regularization
Outputs logits for num_classes actions

# Dataset Structure 
Dataset should be organized in an HMDB51-style format:
data/
 └── HMDB51_split/
      ├── train/
      │    ├── jump/
      │    ├── walk/
      │    ├── run/
      │    └── sit/
      └── val/
           ├── jump/
           ├── walk/
           ├── run/
           └── sit/

