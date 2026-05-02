import random
import shutil
from pathlib import Path

random.seed(42)

source_dir = Path("NCT-CRC-HE-100K")
dest_dir = Path("dataset_split")

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

image_extensions = ["*.tif", "*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.TIF"]

classes = [folder.name for folder in source_dir.iterdir() if folder.is_dir()]

for cls in classes:
    class_folder = source_dir / cls
    images = []

    for ext in image_extensions:
        images.extend(class_folder.rglob(ext))

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split_name, split_imgs in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        split_class_dir = dest_dir / split_name / cls
        split_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            shutil.copy(img_path, split_class_dir / img_path.name)

    print(f"{cls} done -> Train:{len(train_imgs)} Val:{len(val_imgs)} Test:{len(test_imgs)}")

print("Dataset split completed.")