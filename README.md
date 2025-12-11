# Activity Recognition using CNN + LSTM (PyTorch)

A lightweight video action recognition implementation using a CNN (MobileNetV2) feature extractor and an LSTM temporal model, implemented in PyTorch.

Each video frame is passed through MobileNetV2 to produce per-frame features (1280-dim), and an LSTM learns temporal dependencies across frame sequences. A final classifier predicts the action class for each video.

---

### 📍 Project overview
This repository demonstrates a simple and modular approach to video action recognition:
- MobileNetV2 (optional pretrained) extracts per-frame features.
- LSTM models temporal information across sequences of frame features.
- Final fully-connected classifier outputs action logits.

This design keeps the spatial and temporal parts decoupled and is efficient for experiments on modest hardware.

### 📍 Features
- MobileNetV2 backbone (optionally pretrained)
- Configurable LSTM: hidden size, number of layers, bidirectional support
- Dropout & FC classifier head
- Dataset loader expecting HMDB51-style extracted frames
- Training & validation loop with checkpoint saving

### 📍 Repository structure
- model.py — CNN (MobileNetV2) + LSTM architecture
- dataset.py — VideoFrameDataset: loads frame sequences and preprocessing
- train.py — Training and validation pipeline (logging, checkpointing)
- best_model.pth — Example saved best model (if present)
- data/ — Expect HMDB51-style split dataset here

### 📍 Dataset format (HMDB51-style)
The dataset should be laid out like this (example):

data/
└── HMDB51_split/

     ├── train/
     │    ├── jump/
     │    │   ├── video_01/
     │    │   │   ├── frame_0001.jpg
     │    │   │   ├── frame_0002.jpg
     │    │   │   └── ...
     │    │   └── video_02/...
     │    ├── walk/
     │    └── ...
     └── val/
          ├── jump/
          └── ...

- Each class directory contains one subdirectory per video.
- Each video directory contains a sequence of extracted frames (JPEG/PNG).
- You can use any frame extraction approach (FFmpeg, OpenCV) to generate frames from videos.

### 📍 Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/JoanneKoshy/Activity-Recognition-using-CNN-LSTM.git
   cd Activity-Recognition-using-CNN-LSTM
   ```

2. Create a virtual environment (recommended) and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows
   pip install --upgrade pip
   pip install torch torchvision
   pip install opencv-python
   ```
   Add any other dependencies you use (e.g., tqdm, numpy) if needed.

### 📍 Preparing the dataset
- Extract frames from your videos (one folder per video).
- Place class folders under train/ and val/ as shown above.
- Ensure consistent frame naming so dataset.py can sort frames (e.g., frame_0001.jpg, frame_0002.jpg, ...).

### 📍 Training
- The training script is train.py. It loads datasets, builds the model, and runs training and validation loops.
- Common parameters to consider (edit train.py or add CLI args if needed):
  - epochs
  - batch size
  - learning rate
  - sequence length (number of frames per sample)
  - pretrained (whether to load ImageNet pretrained MobileNetV2)
  - freeze_cnn (freeze backbone weights)
  - LSTM hidden size, layers, bidirectional
  - device (cpu / cuda)

Example (illustrative — check train.py for exact arguments):
```bash
python train.py \
  --data-dir data/HMDB51_split \
  --epochs 50 \
  --batch-size 8 \
  --seq-len 16 \
  --lr 1e-4 \
  --pretrained True \
  --freeze-cnn False
```

### 📍 Validation & checkpoints
- The training loop performs validation each epoch and saves the best model as `best_model.pth`.
- Keep an eye on validation accuracy to avoid overfitting; consider early stopping or saving multiple checkpoints.

### 📍 Inference example
Below is a minimal example showing how to load the saved model and run inference on a list of frames. Adapt paths and tensor preprocessing to match dataset.py and model.py preprocessing.

```python
import torch
from model import ActionRecognitionModel  # or the class name in model.py
from torchvision import transforms
from PIL import Image
import numpy as np

# Load model (adjust class name & args as needed)
model = ActionRecognitionModel(num_classes=NUM_CLASSES, pretrained=False, ...)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# Example: load and preprocess frames into a tensor of shape (1, seq_len, C, H, W)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

frame_paths = ["video_01/frame_0001.jpg", "video_01/frame_0002.jpg", ...]
frames = [preprocess(Image.open(p).convert('RGB')) for p in frame_paths]
frames = torch.stack(frames)            # (seq_len, C, H, W)
frames = frames.unsqueeze(0)            # (1, seq_len, C, H, W) if model expects batch-first

with torch.no_grad():
    logits = model(frames)              # adapt based on model input signature
    preds = torch.argmax(logits, dim=1)
    print("Predicted class:", preds.item())
```

### 📍 Notes:
- The exact input shape expected by the model depends on how model.py and dataset.py arrange batch/time/channel dims. Inspect those files and adjust accordingly.

#### Tips & troubleshooting
- If training is slow / memory constrained:
  - Reduce batch size or sequence length.
  - Freeze the CNN backbone (freeze_cnn = True) and only train LSTM + classifier.
  - Use smaller image sizes (e.g., 160x160) if acceptable.
- Use mixed precision training (torch.cuda.amp) for speed & memory improvements.
- Frame sampling: If videos are long, sample evenly spaced frames or use sliding windows.
- Augmentations: random crop, flips, color jitter on frames can help generalization.
- If low accuracy: check dataset balance, ensure labels align to folders, confirm frames are in correct order.

#### Extending the project (ideas)
- Add a dedicated inference script (e.g., infer.py) to process raw video (use OpenCV/ffmpeg to sample frames).
- Add command-line argument parsing for train.py (argparse) to make experiments reproducible.
- Add logging (TensorBoard or Weights & Biases) for visualizing metrics.
- Support more backbones (ResNet, EfficientNet) or replace LSTM with Transformer-based temporal model.
- Add unit tests and a small synthetic dataset for CI.

### 📍 Contributing
Contributions are welcome. Please open issues or PRs for bug fixes, feature requests, or improvements (e.g., adding CLI args, notebooks, or Docker support).

### 📍 License
This project is provided under the MIT License. See LICENSE file for details (or add an appropriate license file).

### 📍 Contact
If you have questions or suggestions, open an issue or contact the repository owner.

---

### 📍 Acknowledgements
- MobileNetV2 backbone is used for efficient feature extraction.
- Inspired by HMDB51-style datasets and common CNN+RNN video recognition pipelines.
