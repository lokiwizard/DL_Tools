import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import zoom

from dataset_utils.animal_utils.animals import get_animal_data_loader
from models.ViT.vit import create_vit

def unnormalize(image):
    mean = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis]
    std = np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]
    image = (image * std) + mean
    return image

def visualize_attention_batch_tensors(images, attn_weights, head=None, avg_heads=True, figsize=(10, 10), cmap='jet', alpha=0.5, unnormalize=None):
    """
    可视化整个批次的注意力热力图，叠加在原始图像上。

    参数：
    - images: 原始图像的张量，形状为 (B, C, H, W)，PyTorch 张量。
    - attn_weights: 注意力权重矩阵，形状为 (B, num_heads, num_patches, num_patches) 的 PyTorch 张量。
    - head: 指定要可视化的注意力头索引（默认为 None）。如果为 None，使用所有头的平均值。
    - avg_heads: 是否对所有注意力头求平均（默认为 True）。如果为 False，需要指定 head。
    - figsize: 单个图像显示的大小（默认为 (10, 10)）。
    - cmap: 热力图的颜色映射（默认为 'jet'）。
    - alpha: 热力图的透明度（默认为 0.5）。
    - unnormalize: 一个函数，用于对图像进行去归一化（可选）。

    返回：
    - 无，直接显示叠加了注意力热力图的图像。
    """
    # 将注意力权重和图像转换为 NumPy 数组
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    batch_size = attn_weights.shape[0]

    # 检查图像数量和批次大小是否一致
    if images.shape[0] != batch_size:
        raise ValueError("图像数量与注意力权重的批次大小不一致。")

    for idx in range(batch_size):
        image = images[idx]  # 形状为 (C, H, W)
        attn = attn_weights[idx]  # 形状为 (num_heads, num_patches, num_patches)

        # 如果需要，取消归一化图像
        if unnormalize is not None:
            image = unnormalize(image)
        else:
            # 假设图像已经在 [0,1] 或 [0,255] 范围内
            # 如果图像进行了归一化（例如 ImageNet 标准化），需要提供 unnormalize 函数
            pass

        # 转换为 [H, W, C]
        image = np.transpose(image, (1, 2, 0))

        # 如果图像是灰度图，将其转换为 RGB
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        # 将像素值从 [0, 1] 范围转换为 [0, 255]
        image = (image * 255).astype(np.uint8)

        # 处理注意力权重
        if head is not None:
            attn_map = attn[head]  # 形状为 (num_patches, num_patches)
        else:
            if avg_heads:
                attn_map = attn.mean(axis=0)  # 对所有头求平均
            else:
                raise ValueError("请指定要可视化的注意力头（head 参数）。")

        # 提取 [CLS] token 的注意力，如果存在的话
        # 假设 [CLS] token 位于第一个位置
        attn_map = attn_map[0, 1:]  # 排除 [CLS] 到自身的注意力
        num_tokens = attn_map.shape[0]
        num_patches = int(np.sqrt(num_tokens))

        # 将一维的注意力权重重塑为二维矩阵
        attn_map = attn_map.reshape(num_patches, num_patches)
        img_height, img_width = image.shape[:2]

        # 使用插值将注意力图调整到与原始图像相同的尺寸
        attn_map_resized = zoom(attn_map, (img_height / num_patches, img_width / num_patches), order=1)
        # 归一化注意力图以适应可视化
        attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min())

        # 创建包含两个子图的图像：原始图像和叠加了热力图的图像
        fig, axs = plt.subplots(1, 2, figsize=figsize)


        # 显示原始图像
        axs[0].imshow(image)
        axs[0].set_title(f"Original Image {idx}")
        axs[0].axis('off')

        # 显示叠加了注意力热力图的图像
        axs[1].imshow(image)
        axs[1].imshow(attn_map_resized, cmap=cmap, alpha=alpha)
        axs[1].set_title(f"Attention Heatmap {idx}")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    train_loader, val_loader, num_classes = get_animal_data_loader(root_dir=r"D:\pyproject\representation_learning_models\dataset_utils\animals",
                                                     batch_size=8,
                                                     image_size=224,
                                                    use_data_augmentation=False)


    images, labels = next(val_loader.__iter__())
    model = create_vit(model_size='base', image_size=224, num_classes=10, drop_path_rate=0.1)

    model.load_state_dict(torch.load(r"D:\pyproject\representation_learning_models\models\trainer\saved_models\vit_base_best_checkpoint.pth"))


    with torch.no_grad():
        model.eval()
        output = model(images)

    print(output.shape)
    # 获取最后一个注意力头的注意力权重
    attn_weights = list(model.attn_dict.values())[-1]
    print(attn_weights.shape)

    visualize_attention_batch_tensors(images, attn_weights, head=None, avg_heads=True, unnormalize=unnormalize)





