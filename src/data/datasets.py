from torch.utils.data import Dataset
from PIL import Image
import torch
from typing import Tuple, Optional, Callable

class Image_Dataset(Dataset):
    """
    A PyTorch Dataset class for loading images and optionally their corresponding labels.

    This class can be used in 'train', 'val', or 'test' mode. In 'train' and 'val' mode, it returns
    both the image and its corresponding label. In 'test' mode, it returns only the image without a label.

    Args:
        mode (str): Mode in which the dataset is used. Must be 'train', 'val', or 'test'.
        image_paths (list[str]): List of file paths for the images.
        labels (Optional[list[int]]): List of labels corresponding to the images. Required in 'train' or 'val' mode, optional in 'test' mode.
        transform (Optional[Callable]): Optional transformation function to apply to the images.

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(index: int): Returns the image and label (for 'train' and 'val') or only the image (for 'test').
    """
    def __init__(
        self,
        mode: str,
        image_paths: list[str],
        labels: Optional[list[int]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode
        self.num_classes = 2

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int] | torch.Tensor:
        img_path = self.image_paths[index]

        image = Image.open(img_path).convert("RGB")

        clean_image_path = img_path.replace("test/", "")

        if self.transform:
            image = self.transform(image)

        if self.mode == "train" or self.mode == "val":
            label = self.labels[index]
            return image, label
        elif self.mode == "test":
            return image, clean_image_path
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'train' or 'val' or 'test'.")
