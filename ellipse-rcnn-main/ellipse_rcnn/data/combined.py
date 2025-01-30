import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import pytorch_lightning as pl
import typer
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader
import torch

from ellipse_rcnn.pl import EllipseRCNNModule
from ellipse_rcnn.data.prasand import PrasadEllipseDataset
from ellipse_rcnn.data.industry import IndustryEllipseDataset
from ellipse_rcnn.data.dataset import DataEllipseDataset
from ellipse_rcnn.data.dataset2 import Data_2_EllipseDataset
from ellipse_rcnn.data.occ24 import occ24EllipseDataset
from ellipse_rcnn.data.occ20 import occ20EllipseDataset
from ellipse_rcnn.data.occ16 import occ16EllipseDataset
from ellipse_rcnn.data.occ12 import occ12EllipseDataset
from ellipse_rcnn.data.occ8 import occ8EllipseDataset
from ellipse_rcnn.data.occ4 import occ4EllipseDataset
from ellipse_rcnn.data.utils import collate_fn  # Ensure this is correctly imported

app = typer.Typer(pretty_exceptions_show_locals=False)

# Mapping dataset names to their respective classes and directories
DATASET_CONFIG = {
    "Industry": {
        "class": IndustryEllipseDataset,
        "images_dir": r"D:\Exercises\HIWI\EllipDet-master\Industrial\images",
        "annotations_dir": r"D:\Exercises\HIWI\EllipDet-master\Industrial\gt",
    },
    "Prasad": {
        "class": PrasadEllipseDataset,
        "images_dir": r"D:\Exercises\HIWI\EllipDet-master\Prasad\Prasad\images",
        "annotations_dir": r"D:\Exercises\HIWI\EllipDet-master\Prasad\Prasad\gt",
    },
    "Data": {
        "class": DataEllipseDataset,
        "images_dir": r"D:\Exercises\HIWI\EllipDet-master\Dataset#1\Dataset#1\images",
        "annotations_dir": r"D:\Exercises\HIWI\EllipDet-master\Dataset#1\Dataset#1\gt",
    },
    "Data_2": {
        "class": Data_2_EllipseDataset,
        "images_dir": r"D:\Exercises\HIWI\EllipDet-master\Dataset#2\Dataset#2\images",
        "annotations_dir": r"D:\Exercises\HIWI\EllipDet-master\Dataset#2\Dataset#2\gt",
    },
    "occ24": {
        "class": occ24EllipseDataset,
        "images_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O24\images",
        "annotations_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O24\gt",
    },
    "occ20": {
        "class": occ20EllipseDataset,
        "images_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O20\images",
        "annotations_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O20\gt",
    },
    "occ16": {
        "class": occ16EllipseDataset,
        "images_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O16\images",
        "annotations_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O16\gt",
    },
    "occ12": {
        "class": occ12EllipseDataset,
        "images_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O12\images",
        "annotations_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O12\gt",
    },
    "occ8": {
        "class": occ8EllipseDataset,
        "images_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O8\images",
        "annotations_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O8\gt",
    },
    "occ4": {
        "class": occ4EllipseDataset,
        "images_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O4\images",
        "annotations_dir": r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O4\gt",
    },
}

class CombinedEllipseDataModule(LightningDataModule):
    def __init__(
        self,
        datasets: List[str],
        batch_size: int,
        num_workers: int,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
        resize: Tuple[int, int] = (640, 640),
        transform=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.resize = resize
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

        for dataset_name in self.datasets:
            if dataset_name not in DATASET_CONFIG:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            config = DATASET_CONFIG[dataset_name]
            dataset_class = config["class"]
            images_dir = config["images_dir"]
            annotations_dir = config["annotations_dir"]

            # List all image and annotation files
            image_files = sorted([
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
            ])
            annotation_files = sorted([
                os.path.join(annotations_dir, f)
                for f in os.listdir(annotations_dir)
                if f.lower().endswith(".txt")
            ])

            if len(image_files) != len(annotation_files):
                raise ValueError(f"Mismatch between images and annotations in dataset {dataset_name}.")

            # Shuffle indices
            random.seed(self.seed)
            indices = list(range(len(image_files)))
            random.shuffle(indices)

            train_end = int(self.train_split * len(indices))
            val_end = train_end + int(self.val_split * len(indices))

            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

            # Create subsets
            train_image_files = [image_files[i] for i in train_indices]
            train_annotation_files = [annotation_files[i] for i in train_indices]

            val_image_files = [image_files[i] for i in val_indices]
            val_annotation_files = [annotation_files[i] for i in val_indices]

            test_image_files = [image_files[i] for i in test_indices]
            test_annotation_files = [annotation_files[i] for i in test_indices]

            # Initialize datasets
            train_dataset = dataset_class(
                image_files=train_image_files,
                annotation_files=train_annotation_files,
                transform=self.transform,
                resize=self.resize,
            )
            val_dataset = dataset_class(
                image_files=val_image_files,
                annotation_files=val_annotation_files,
                transform=self.transform,
                resize=self.resize,
            )
            test_dataset = dataset_class(
                image_files=test_image_files,
                annotation_files=test_annotation_files,
                transform=self.transform,
                resize=self.resize,
            )

            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)

        # Combine the datasets
        self.train_dataset = ConcatDataset(self.train_datasets) if self.train_datasets else None
        print(f"train: {self.train_dataset}")
        self.val_dataset = ConcatDataset(self.val_datasets) if self.val_datasets else None
        print(f'val: {self.val_dataset}')
        self.test_dataset = ConcatDataset(self.test_datasets) if self.test_datasets else None
        print(f'test: {self.test_dataset}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,  # Ensure collate_fn is compatible
        ) if self.train_dataset else None

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        ) if self.val_dataset else None

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        ) if self.test_dataset else None

@app.command()
def train_model(
    iterations: int = typer.Option(1, help="Number of iterations to train the model."),
    lr: float | None = typer.Option(
        None, help="Learning rate value. Disables lr sampling."
    ),
    weight_decay: float | None = typer.Option(
        None, help="Weight decay value. Disables weight_decay sampling."
    ),
    lr_min: float = typer.Option(1e-5, help="Minimum learning rate for sampling."),
    lr_max: float = typer.Option(1e-3, help="Maximum learning rate for sampling."),
    weight_decay_min: float = typer.Option(
        1e-5, help="Minimum weight decay for sampling."
    ),
    weight_decay_max: float = typer.Option(
        1e-3, help="Maximum weight decay for sampling."
    ),
    num_workers: int = typer.Option(4, help="Number of workers for data loading."),
    batch_size: int = typer.Option(16, help="Batch size for training."),
    datasets: List[str] = typer.Option(
        ["Industry", "Prasad", "Data", "Data_2", "occ24", "occ20", "occ16", "occ12", "occ8", "occ4"],
        help="Datasets to use for training. Provide multiple datasets as separate options.",
    ),
    accelerator: str = typer.Option("auto", help="Type of accelerator to use."),
) -> None:
    if iterations > 1 and (lr is not None or weight_decay is not None):
        print(
            "Warning: Running with multiple iterations with a fixed learning rate or weight decay."
        )

    print(f"num-workers: {num_workers}, batch-size: {batch_size}")
    for iteration in range(iterations):
        sampled_lr = random.uniform(lr_min, lr_max)
        sampled_weight_decay = random.uniform(weight_decay_min, weight_decay_max)
        current_lr = lr if lr is not None else sampled_lr
        current_weight_decay = (
            weight_decay if weight_decay is not None else sampled_weight_decay
        )

        print(f"Using parameters - Learning rate: {current_lr}, Weight decay: {current_weight_decay}")
        print(f"Starting iteration {iteration + 1}/{iterations}")
        pl_module = EllipseRCNNModule(lr=current_lr, weight_decay=current_weight_decay)

        # Use the CombinedEllipseDataModule
        datamodule = CombinedEllipseDataModule(
            datasets=datasets,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Setup the data module
        datamodule.setup()

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss_total",
            dirpath="checkpoints",
            filename=r"loss={val/loss_total:.5f}-e={epoch:02d}",
            auto_insert_metric_name=False,
            save_top_k=1,
            mode="min",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val/loss_total",
            patience=5,
            mode="min",
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            precision=32,  # Changed from "32-true" to 32
            max_epochs=40,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

        trainer.fit(pl_module, datamodule=datamodule)
        print(f"Completed iteration {iteration + 1}/{iterations}")

if __name__ == "__main__":
    app()