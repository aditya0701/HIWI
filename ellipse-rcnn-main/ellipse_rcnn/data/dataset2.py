import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import pytorch_lightning as pl
import math

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
from ellipse_rcnn.data.utils import collate_fn


class Data_2_EllipseDataset(Dataset):
    """
    Dataset class for industrial ellipse data.
    """

    def __init__(self, image_files: list, annotation_files: list, transform=None,  resize=(640, 640)):
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
        self.resize = resize

    def __getitem__(self, idx) -> ImageTargetTuple:
        """
        Retrieve an image and its corresponding annotations.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, target), where:
                - image is a transformed PIL image.
                - target is a dictionary containing ellipse and bounding box information.
        """
        print(f"Processing sample {idx}")
        image_path = self.image_files[idx]
        annotation_path = self.annotation_files[idx]

        # Load image
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        original_height, original_width = image.shape[:2]
        target_size = self.resize
        
        print(f"original size: {original_width} x {original_height}")
        image = cv2.resize(image, (target_size[0], target_size[1]))
        print(f"target size_1: {image.shape[1]} x {image.shape[0]}")
        transform = transforms.ToTensor()
        image = transform(image)

        if self.transform:
            image = self.transform(image)

        scale_x = target_size[0] / original_width
        scale_y = target_size[1] / original_height

        with open(annotation_path, "r") as f:
            num_objs = int(f.readline().strip())
            
            cx_list, cy_list, a_list, b_list, theta_list = [], [], [], [], []
            for _ in range(num_objs):
                line = f.readline().strip()
                if not line:  
                    continue    
                
                try:
                    x_center, y_center, width, height, angle = map(float, line.strip().split())
                    # print(f"Ellipse parameters: x={x_center}, y={y_center}, a={width}, b={height}, angle={angle}")
                    if width <= 0 or height <= 0:
                        raise ValueError(f"Invalid ellipse axes: a={width}, b={height}")
                    x_center = int(x_center * scale_x)
                    y_center = int(y_center * scale_y)
                    width = int(width * scale_x)
                    height = int(height * scale_y)
                    
                    original_angle_rad = math.radians(angle)
                    tan_2theta = math.tan(2 * original_angle_rad)
                    scale_ratio = scale_y / scale_x
                    tan_2theta_prime = (tan_2theta * scale_ratio) / (1 + tan_2theta**2 * (scale_ratio**2 - 1))
                    angle = 0.5 * math.atan(tan_2theta_prime)
                    # new_angle = math.degrees(new_angle_rad)

                    a_list.append(width)
                    b_list.append(height)
                    cx_list.append(x_center)
                    cy_list.append(y_center)
                    theta_list.append(angle)
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
        boxes = bbox_ellipse(ellipses)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Filter out ellipses with very small areas
        valid_indices = area > 1e-1
        boxes = boxes[valid_indices]
        area = area[valid_indices]
        ellipses = ellipses[valid_indices]

        labels = torch.ones((num_objs,), dtype=torch.int64)  # Assuming all objects belong to class 1
        image_id = torch.tensor([idx], dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Create the target dictionary
        target = TargetDict(
            boxes=boxes,
            labels=labels,
            image_id=image_id,
            area=area,
            iscrowd=iscrowd,
            ellipse_params=ellipses,
        )
        # print("Sample targets:", target)
        return image, target

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.image_files)



class Data_2_EllipseDataModule(pl.LightningDataModule):
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

        self.train_dataset = Data_2_EllipseDataset(
            [image_files[i] for i in train_indices],
            [annotation_files[i] for i in train_indices],
            transform=self.transform
        )
        self.val_dataset = Data_2_EllipseDataset(
            [image_files[i] for i in val_indices],
            [annotation_files[i] for i in val_indices],
            transform=self.transform
        )
        self.test_dataset = Data_2_EllipseDataset(
            [image_files[i] for i in test_indices],
            [annotation_files[i] for i in test_indices],
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          collate_fn=collate_fn, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          collate_fn=collate_fn,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          collate_fn=collate_fn,
                          num_workers=self.num_workers)