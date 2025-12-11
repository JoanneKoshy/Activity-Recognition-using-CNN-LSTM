import os
import random
import shutil

SOURCE = "data/HMDB51_small"
TARGET = "data/HMDB51_split"

TRAIN_RATIO = 0.8

for split in ["train", "val"]:
    os.makedirs(os.path.join(TARGET, split), exist_ok=True)

for cls in sorted(os.listdir(SOURCE)):
    cls_path = os.path.join(SOURCE, cls)
    if not os.path.isdir(cls_path):
        continue

    # Each subfolder = one video
    videos = [v for v in os.listdir(cls_path) 
              if os.path.isdir(os.path.join(cls_path, v))]

    random.shuffle(videos)
    split_idx = int(len(videos) * TRAIN_RATIO)

    train_videos = videos[:split_idx]
    val_videos = videos[split_idx:]

    # create class folders
    os.makedirs(os.path.join(TARGET, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(TARGET, "val", cls), exist_ok=True)

    # copy folders
    for v in train_videos:
        shutil.copytree(
            os.path.join(cls_path, v),
            os.path.join(TARGET, "train", cls, v),
            dirs_exist_ok=True
        )

    for v in val_videos:
        shutil.copytree(
            os.path.join(cls_path, v),
            os.path.join(TARGET, "val", cls, v),
            dirs_exist_ok=True
        )

    print(f"{cls} -> Train: {len(train_videos)}, Val: {len(val_videos)}")

print("\n🎉 DONE: Frame-based split created in data/HMDB51_split")
