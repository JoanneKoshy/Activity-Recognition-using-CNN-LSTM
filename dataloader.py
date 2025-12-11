from dataset import VideoFrameDataset
from transforms import get_transforms
import torch

train_path = "data/HMDB51_split/train"
val_path = "data/HMDB51_split/val"

transform = get_transforms()

train_ds = VideoFrameDataset(train_path, num_frames=16, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True)

frames, label = next(iter(train_loader))

print("Frames shape:", frames.shape)
print("Label:", label)
