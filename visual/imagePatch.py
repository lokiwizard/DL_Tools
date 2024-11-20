# 工具说明

"""
This tool is used for patchifying and unpatchifying images. It supports visualization and saving the results.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Union
from einops import rearrange

def read_image(image_path: str) -> Tuple[Image.Image, int]:
    """
    Read an image.

    Args:
        image_path (str): The path to the image.

    Returns:
        PIL.Image: The image.
    """
    assert os.path.exists(image_path), "The image path does not exist."
    image = Image.open(image_path)
    num_channels = 3  # default value
    # get num_channels
    if image.mode == "RGB":
        num_channels = 3
    elif image.mode == "L":
        num_channels = 1
    elif image.mode == "RGBA":
        num_channels = 4
    else:
        raise ValueError("The image mode is not supported.")
    return image, num_channels

def patchify(image_path: str, patch_size: Union[int, tuple], image_size: Union[int, tuple]) -> np.ndarray:
    """
    Patchify an image.

    Args:
        image_path (str): The path to the image.
        patch_size (int): The size of the patch.

    Returns:
        torch.Tensor: The patchified image.
    """
    image, num_channels = read_image(image_path)

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, "The image size must be divisible by the patch size."

    # resize the image
    image = image.resize(image_size)
    image = np.array(image)  # H x W x C

    # get the shape of the image and the patch
    H, W = image_size[0], image_size[1]
    hp, wp = patch_size[0], patch_size[1]

    # get the number of patches
    num_patches = (H//hp) * (W//wp)

    # use einops to rearrange the image
    patches = rearrange(image, '(nh hp) (nw wp) c -> (nh nw) hp wp c ', hp=hp, wp=wp, c=num_channels, nh=H//hp, nw=W//wp)

    return patches

def unpatchify(patches, image_size: Union[int, tuple]) -> Image.Image:
    """
    Unpatchify an image.

    Args:
        patches (torch.Tensor): The patchified image.
        image_size (int): The size of the image.

    Returns:
        PIL.Image: The unpatchified image.
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    # get the shape of the image and the patch
    H, W = image_size[0], image_size[1]
    hp, wp = patches.shape[1], patches.shape[2]

    # use einops to rearrange the image
    image = rearrange(patches, '(nh nw) hp wp c -> (nh hp) (nw wp) c', hp=hp, wp=wp, c=patches.shape[3], nh=H//hp, nw=W//wp)
    image = Image.fromarray(image)
    return image

def visualize_patches(patches: np.ndarray, is_save: bool = False, save_path: str = None) -> None:
    """
    Visualize the patches.

    Args:
        patches (torch.Tensor): The patchified image.
        num_rows (int): The number of rows in the visualization.
        num_cols (int): The number of columns in the visualization.
    """
    num_patches = patches.shape[0]
    num_rows = num_cols = int(np.sqrt(num_patches))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))
    for i in range(num_rows):
        for j in range(num_cols):
            patch = patches[i*num_cols + j]
            axes[i, j].imshow(patch)
            axes[i, j].axis("off")
            # save each patch
            if is_save:
                plt.imsave(save_path
                            + f"/patch_{i*num_cols + j}.jpg", patch)
    plt.show()



if __name__ == "__main__":
    patches = patchify("./cat.jpeg", 32, 256)
    print(patches.shape)
    visualize_patches(patches)


