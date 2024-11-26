import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# 为训练集和验证集创建自定义的 Dataset，以应用不同的 transform
# 下载地址：https://www.kaggle.com/datasets/alessiocorrado99/animals10
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


def get_animal_data_loader(root_dir, batch_size=32, test_size=0.2,
                           random_state=42, use_data_augmentation=True, image_size=224,
                           num_workers=4):

    # Define the transformations

    if use_data_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    # Load the dataset
    dataset = datasets.ImageFolder(root=root_dir)

    # get labels
    labels = [label for _, label in dataset]

    # Split the dataset into training and test sets
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=random_state, stratify=labels)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create the CustomDataset objects
    train_dataset = CustomDataset(train_dataset, transform=train_transform)
    val_dataset = CustomDataset(val_dataset, transform=val_transform)

    # Create the DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    num_classes = len(dataset.classes)

    return train_loader, val_loader, num_classes



