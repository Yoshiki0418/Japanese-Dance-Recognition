from glob import glob
import os
from typing import Tuple

def get_image_paths_and_labels(base_path: str) -> Tuple[list[str], list[int]] | list[str]:
    """
    Args:
        base_path (str): The base directory containing the 'hold' and 'non-hold' subdirectories,
                         or just the image files for testing.

    Returns:
        Union[Tuple[List[str], List[int]], List[str]]:
            - If base_path is "train", returns a tuple containing:
                - list of image file paths (str)
                - list of corresponding labels (int)
            - Otherwise, returns a list of image file paths (str) for test data.
    """
    if base_path == "train":
        hold_dir = os.path.join(base_path, "hold")
        non_hold_dir = os.path.join(base_path, "not-hold")

        hold_images = glob(os.path.join(hold_dir, "*.jpg"))
        non_hold_images = glob(os.path.join(non_hold_dir, "*.jpg"))

        image_paths = hold_images + non_hold_images
        labels = [1] * len(hold_images) + [0] * len(non_hold_images)

        return image_paths, labels

    else:
        test_images = glob(os.path.join(base_path, "*.jpg"))

        return test_images

    