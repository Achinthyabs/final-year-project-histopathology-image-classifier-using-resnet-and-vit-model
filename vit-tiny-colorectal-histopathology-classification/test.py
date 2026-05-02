from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dir = "dataset_split/train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

print("Classes:", train_dataset.classes)
print("Total training images:", len(train_dataset))

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch labels:", labels)