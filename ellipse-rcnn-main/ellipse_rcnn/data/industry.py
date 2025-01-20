import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

from ellipse_rcnn.core.types import ImageTargetTuple, TargetDict
from ellipse_rcnn.core.ops import (
    ellipse_axes,
    ellipse_center,
    ellipse_angle,
    bbox_ellipse,
)
from torch.utils.data import Dataset
import os
import torch
from PIL import Image


class IndustryEllipseDataset(Dataset):
    """
    Dataset class for industrial ellipse data.
    """

    def __init__(self, image_files: list, annotation_files: list, transform=None):
        """
        Initialize the dataset.

        Args:
            image_files (list): List of image file paths.
            annotation_files (list): List of annotation file paths.
            transform: Optional transform to be applied to the images.
        """
        self.image_files = image_files
        self.annotation_files = annotation_files
        self.transform = transform

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding annotations.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, target), where:
                - image is a transformed PIL image.
                - target is a dictionary containing ellipse and bounding box information.
        """
        # self.image_path = self.image_files[idx]
        # self.annotation_path = self.annotation_files[idx]
        image_path = self.image_files[idx]
        annotation_path = self.annotation_files[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load annotations
        with open(annotation_path, "r") as f:
            lines = f.readlines()

            # First line specifies the number of ellipses
            num_objs = int(lines[0].strip())
            if len(lines) - 1 != num_objs:
                raise ValueError("Mismatch between number of objects and data lines.")

            # Parse ellipses from remaining lines
            cx_list, cy_list, a_list, b_list, theta_list = [], [], [], [], []
            for line in lines[1:]:
                try:
                    cx, cy, a, b, theta = map(float, line.strip().split())
                    if a <= 0 or b <= 0:
                        raise ValueError(f"Invalid ellipse axes: a={a}, b={b}")
                    a_list.append(a)
                    b_list.append(b)
                    cx_list.append(cx)
                    cy_list.append(cy)
                    theta_list.append(theta)
                except ValueError as e:
                    raise ValueError(f"Error processing line: {line.strip()}") from e
            # Create stacked tensor 
            a = torch.tensor(a_list)
            b = torch.tensor(b_list)
            cx = torch.tensor(cx_list)
            cy = torch.tensor(cy_list)
            theta = torch.tensor(theta_list)

            ellipses = torch.stack([a, b, cx, cy, theta], dim=-1).reshape(-1, 5)
        # Compute bounding boxes using ellipse parameters
        boxes = bbox_ellipse(ellipses).reshape(-1, 4)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Filter out ellipses with very small areas
        valid_indices = area > 1e-1
        boxes = boxes[valid_indices]
        area = area[valid_indices]
        ellipses = ellipses[valid_indices]

        num_objs = len(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # Assuming all objects belong to class 1
        image_id = torch.tensor([idx], dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Create the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "ellipse_params": ellipses,
        }

        return image, target

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.image_files)



class IndustryEllipseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        batch_size: int,
        num_workers: int = 4,
        transform=None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        # Get all image and annotation file paths
        image_files = sorted([
            os.path.join(self.images_dir, f)
            for f in os.listdir(self.images_dir)
            if f.endswith(".jpg") or f.endswith(".bmp")  # Include both .jpg and .bmp files
        ])

        annotation_files = sorted([
            os.path.join(self.annotations_dir, f)
            for f in os.listdir(self.annotations_dir)
            if f.endswith(".txt")
        ])
        
        # Ensure alignment between images and annotations
        assert len(image_files) == len(annotation_files), "Mismatch between images and annotations."
        print(image_files)
        # Split into training, validation, and test sets
        dataset_size = len(image_files)
        train_size = int(self.train_split * dataset_size)
        val_size = int(self.val_split * dataset_size)

        # Shuffle and split
        random.seed(self.seed)
        indices = list(range(dataset_size))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        self.train_dataset = IndustryEllipseDataset(
            [image_files[i] for i in train_indices],
            [annotation_files[i] for i in train_indices],
            transform=self.transform
        )
        self.val_dataset = IndustryEllipseDataset(
            [image_files[i] for i in val_indices],
            [annotation_files[i] for i in val_indices],
            transform=self.transform
        )
        self.test_dataset = IndustryEllipseDataset(
            [image_files[i] for i in test_indices],
            [annotation_files[i] for i in test_indices],
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)