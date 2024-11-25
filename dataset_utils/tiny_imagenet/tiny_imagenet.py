import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from typing import Tuple, List
from PIL import Image

BASE_DATA_DIR = "../tiny-imagenet-200/"  # 数据集存放路径，需要根据实际情况修改

# 图像增强函数

def train_transform(image_size=64):

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # resize
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(64, padding=4),  # 随机裁剪（保持分辨率）
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])

    return train_transforms

def val_transform(image_size=64):

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return val_transforms

class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir: str, annotations_file: str, class_to_idx: dict, transform=None):
        """
        Tiny ImageNet 验证集数据集类（标签索引与训练集对齐）
        :param val_dir: 验证集图片目录
        :param annotations_file: 验证集标签文件路径
        :param class_to_idx: 训练集的类别到索引的映射
        :param transform: 图像变换
        """
        self.val_dir = val_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

        # 读取 val_annotations.txt
        self.image_labels = []
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                file_name, class_id = parts[0], parts[1]
                if class_id in class_to_idx:
                    self.image_labels.append((file_name, class_to_idx[class_id]))

    def __len__(self) -> int:
        return len(self.image_labels)

    def __getitem__(self, idx: int) -> Tuple:
        """
        获取图像和标签
        :param idx: 索引
        :return: (图像, 标签)
        """
        file_name, label = self.image_labels[idx]
        img_path = os.path.join(self.val_dir, file_name)
        image = Image.open(img_path).convert('RGB')  # 打开图像

        if self.transform:
            image = self.transform(image)

        return image, label

def load_tiny_imagenet(use_data_augment:bool, batch_size:int, num_workers:int=1, image_size=64) -> Tuple[DataLoader, DataLoader]:
    """
    加载Tiny ImageNet数据集
    :param use_data_augment: 是否使用数据增强
    :param batch_size: 批大小
    :param num_workers: 线程数
    :return: 训练集、验证集
    """


    train_transforms = train_transform(image_size=image_size) if use_data_augment else val_transform(image_size=image_size)
    val_transforms = val_transform(image_size=image_size)

    train_dataset = datasets.ImageFolder(os.path.join(BASE_DATA_DIR, "train"), transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 验证集加载，使用训练集的 class_to_idx 映射
    val_dataset = TinyImageNetValDataset(
        val_dir=os.path.join(BASE_DATA_DIR, "val/images"),
        annotations_file=os.path.join(BASE_DATA_DIR, "val/val_annotations.txt"),
        class_to_idx=train_dataset.class_to_idx,
        transform=val_transforms
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return train_loader, val_loader

# use case
"""
if __name__ == '__main__':
    train_loader, val_loader = load_tiny_imagenet(use_data_augment=True, batch_size=64, num_workers=1, image_size=64)
    for x, y in val_loader:
        print(x.shape, y.shape)
        break
"""






