# model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    """
    MobileNetV2 backbone that outputs a feature vector per frame.
    By default we remove the classifier and use global avgpool to get a 1280-D vector.
    """
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        mob = models.mobilenet_v2(pretrained=pretrained)
        # Use the feature layers
        self.features = mob.features  # contains conv layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # to get (B, 1280, 1, 1)
        self.out_dim = 1280

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: (B*T, 3, H, W)
        returns: (B*T, out_dim)
        """
        x = self.features(x)                # (B*T, C, h, w)
        x = self.pool(x)                    # (B*T, C, 1, 1)
        x = x.view(x.size(0), -1)           # (B*T, C)
        return x


class CNN_LSTM_ActivityRecognizer(nn.Module):
    """
    Full model: per-frame CNN -> LSTM over time -> classifier.
    Input: frames tensor shaped (B, T, 3, H, W)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        num_classes,
        lstm_hidden=256,
        lstm_layers=1,
        bidirectional=False,
        dropout=0.3,
        pretrained_cnn=True,
        freeze_cnn=True
    ):
        super().__init__()
        self.cnn = CNNFeatureExtractor(pretrained=pretrained_cnn, freeze_backbone=freeze_cnn)
        feat_dim = self.cnn.out_dim

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden * self.num_directions, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, frames):
        """
        frames: (B, T, 3, H, W)
        returns: logits (B, num_classes)
        """
        B, T, C, H, W = frames.shape

        # reshape to process all frames at once through CNN
        x = frames.view(B * T, C, H, W)            # (B*T, 3, H, W)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.cnn.parameters())):
            feats = self.cnn(x)                   # (B*T, feat_dim)

        feats = feats.view(B, T, -1)               # (B, T, feat_dim)

        # LSTM expects (B, T, feat_dim) with batch_first=True
        lstm_out, (h_n, c_n) = self.lstm(feats)    # lstm_out: (B, T, hidden * num_directions)

        # Get last relevant hidden: use last time-step output
        if self.bidirectional:
            # concatenate last forward and backward hidden (from output)
            last_out = lstm_out[:, -1, :]        # (B, hidden * 2)
        else:
            last_out = lstm_out[:, -1, :]        # (B, hidden)

        logits = self.classifier(last_out)        # (B, num_classes)
        return logits
