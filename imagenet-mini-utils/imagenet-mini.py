import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 下载地址：https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000

def get_imagenet_mini_loader(root_dir, batch_size, use_data_augmentation, image_size, num_workers):

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

    train_dataset = datasets.ImageFolder(root=root_dir + '/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root=root_dir + '/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    num_classes = len(train_dataset.classes)

    return train_loader, val_loader, num_classes

if __name__ == "__main__":

    root_dir = r'D:\pyproject\representation_learning_models\dataset_utils\imagenet-mini'
    batch_size = 32
    use_data_augmentation = True
    image_size = 224
    num_workers = 4

    train_loader, val_loader, num_classes = get_imagenet_mini_loader(root_dir, batch_size, use_data_augmentation, image_size, num_workers)


    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i} - images: {images.size()}, labels: {labels.size()}")
        if i == 0:
            break

