import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform

        self.samples = []  # (video_path, class_idx)

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # Collect videos
        for cls in classes:
            cls_path = os.path.join(root_dir, cls)

            for video in os.listdir(cls_path):
                video_path = os.path.join(cls_path, video)
                if os.path.isdir(video_path):
                    self.samples.append((video_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def load_frames(self, video_path):
        frames = sorted([
            os.path.join(video_path, f)
            for f in os.listdir(video_path)
            if f.endswith(".jpg")
        ])

        # Randomly sample N frames
        if len(frames) >= self.num_frames:
            start = random.randint(0, len(frames) - self.num_frames)
            frames = frames[start:start + self.num_frames]
        else:
            # Pad if not enough frames
            frames = frames + [frames[-1]] * (self.num_frames - len(frames))

        imgs = [Image.open(f).convert("RGB") for f in frames]
        return imgs

    def __getitem__(self, idx):
        video_path, class_idx = self.samples[idx]
        frames = self.load_frames(video_path)

        if self.transform:
            frames = [self.transform(f) for f in frames]

        frames = torch.stack(frames)  # [N, 3, H, W]
        return frames, class_idx
