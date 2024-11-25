import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat


class OxfordFlowersDataset(Dataset):
    def __init__(self, root, split='train', transform=None, image_size=(224, 224)):
        self.root = root
        self.split = split
        self.image_size = image_size

        # 加载图像标签和分割信息
        labels = loadmat(os.path.join(root, 'imagelabels.mat'))['labels'][0]
        setids = loadmat(os.path.join(root, 'setid.mat'))

        # 获取数据集分割
        if split == 'train':
            image_indices = setids['trnid'][0] - 1
        elif split == 'val':
            image_indices = setids['valid'][0] - 1
        else:  # test
            image_indices = setids['tstid'][0] - 1

        # 准备图像路径和标签
        self.image_paths = [
            os.path.join(root, 'jpg', f'image_{idx + 1:05d}.jpg')
            for idx in image_indices
        ]
        # 在加载数据集时，将标签转换为torch.long类型
        self.labels = torch.tensor([labels[idx] - 1 for idx in image_indices], dtype=torch.long)

        # 设置变换
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_oxford_flowers_loaders(
        root,
        batch_size=32,
        num_workers=4,
        image_size=(224, 224),
        use_augmentation=True
):
    # 训练集数据增强
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
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
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 创建数据集
    train_dataset = OxfordFlowersDataset(
        root,
        split='train',
        transform=train_transform,
        image_size=image_size
    )
    val_dataset = OxfordFlowersDataset(
        root,
        split='val',
        transform=val_transform,
        image_size=image_size
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, len(set(train_dataset.labels))

# 使用示例
if __name__ == "__main__":
    """
    oxford_flowers_dataset/
├── jpg/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── imagelabels.mat
└── setid.mat
    
    """
    train_loader, val_loader, num_classes = get_oxford_flowers_loaders(
    root=r'D:\pyproject\representation_learning_models\dataset_utils\102flowers',
    image_size=(224, 224),  # 可自定义图像大小
    use_augmentation=True   # 控制是否使用数据增强
 )

    for images, labels in val_loader:
        print(images.shape, labels.shape)
        break