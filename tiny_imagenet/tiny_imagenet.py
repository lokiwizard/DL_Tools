import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        参数:
        root_dir: Tiny ImageNet数据集的根目录
        mode: 'train' 或 'val'
        transform: 图像预处理转换
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            # 训练集目录结构: root_dir/train/n01443537/images/n01443537_0.JPEG
            self.data = []
            self.targets = []
            train_dir = os.path.join(root_dir, 'train')
            self.classes = sorted(os.listdir(train_dir))

            # 为每个类别创建一个数字标签
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

            # 收集所有图像路径和标签
            for class_name in self.classes:
                class_dir = os.path.join(train_dir, class_name, 'images')
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.JPEG'):
                            self.data.append(os.path.join(class_dir, img_name))
                            self.targets.append(self.class_to_idx[class_name])

        elif mode == 'val':
            # 验证集目录结构: root_dir/val/images/val_0.JPEG
            # 读取验证集标签文件
            val_annotations_file = os.path.join(root_dir, 'val', 'val_annotations.txt')
            df = pd.read_csv(val_annotations_file, sep='\t', header=None,
                             names=['filename', 'class', 'x', 'y', 'w', 'h'])

            # 获取类别到索引的映射
            train_dir = os.path.join(root_dir, 'train')
            self.classes = sorted(os.listdir(train_dir))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

            self.data = []
            self.targets = []
            val_img_dir = os.path.join(root_dir, 'val', 'images')

            for idx, row in df.iterrows():
                img_path = os.path.join(val_img_dir, row['filename'])
                if os.path.exists(img_path):
                    self.data.append(img_path)
                    self.targets.append(self.class_to_idx[row['class']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]

        # 读取图像
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target


def get_tiny_imagenet_loaders(root_dir, image_size=64, batch_size=128, num_workers=4):
    """
    创建Tiny ImageNet的训练和验证数据加载器

    参数:
    root_dir: Tiny ImageNet数据集的根目录
    image_size: 输出图像的大小（默认64）
    batch_size: 批次大小
    num_workers: 数据加载的工作进程数

    返回:
    train_loader, val_loader
    """
    # 定义数据预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.143)),  # 稍大一些以便进行中心裁剪
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = TinyImageNetDataset(root_dir, mode='train', transform=train_transform)
    val_dataset = TinyImageNetDataset(root_dir, mode='val', transform=val_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    num_classes = len(train_dataset.classes)
    return train_loader, val_loader, num_classes


