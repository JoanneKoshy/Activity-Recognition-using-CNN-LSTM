import torch
from model import CNN_LSTM_ActivityRecognizer

num_classes = 20   # replace with number of classes you have
model = CNN_LSTM_ActivityRecognizer(num_classes=num_classes, pretrained_cnn=True, freeze_cnn=True)

# tiny dummy batch from your dataloader shape: [B, T, 3, H, W]
B, T, C, H, W = 2, 16, 3, 112, 112
dummy = torch.randn(B, T, C, H, W)

logits = model(dummy)   # (B, num_classes)
print("logits shape:", logits.shape)
