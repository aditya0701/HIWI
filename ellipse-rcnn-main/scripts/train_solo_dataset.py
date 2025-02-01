import random

import pytorch_lightning as pl
import typer
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from ellipse_rcnn.pl import EllipseRCNNModule
from ellipse_rcnn.data.prasand import PrasadEllipseDataModule
from ellipse_rcnn.data.industry import IndustryEllipseDataModule
from ellipse_rcnn.data.dataset import DataEllipseDataModule
from ellipse_rcnn.data.dataset2 import Data_2_EllipseDataModule
from ellipse_rcnn.data.occ24 import occ24EllipseDataModule
from ellipse_rcnn.data.occ20 import occ20EllipseDataModule
from ellipse_rcnn.data.occ16 import occ16EllipseDataModule
from ellipse_rcnn.data.occ12 import occ12EllipseDataModule
from ellipse_rcnn.data.occ8 import occ8EllipseDataModule
from ellipse_rcnn.data.occ4 import occ4EllipseDataModule

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train_model(
    iterations: int = typer.Option(3, help="Number of iterations to train the model."),
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
    dataset: str = typer.Option("Prasad", help="Dataset to use for training."),
    accelerator: str = typer.Option("auto", help="Type of accelerator to use."),
) -> None:
    datamodule: LightningDataModule
    match dataset:
        case "Industry":
            datamodule = IndustryEllipseDataModule(
                r"D:\Exercises\HIWI\EllipDet-master\Industrial\images",
                r"D:\Exercises\HIWI\EllipDet-master\Industrial\gt", 
                num_workers=num_workers, 
                batch_size=batch_size
            )

        case "Prasad":
            datamodule = PrasadEllipseDataModule(
                r"D:\Exercises\HIWI\EllipDet-master\Final_dataset\images",
                r"D:\Exercises\HIWI\EllipDet-master\Final_dataset\gt",
                batch_size=batch_size,
                num_workers=num_workers,
            )
        case "Data":
            datamodule = DataEllipseDataModule(
            r"D:\Exercises\HIWI\EllipDet-master\Dataset#1\Dataset#1\images",
            r"D:\Exercises\HIWI\EllipDet-master\Dataset#1\Dataset#1\gt",
            batch_size=batch_size,
            num_workers=num_workers,
            )

        case "Data_2":
            datamodule = Data_2_EllipseDataModule(
            r"D:\Exercises\HIWI\EllipDet-master\Dataset#2\Dataset#2\images",
            r"D:\Exercises\HIWI\EllipDet-master\Dataset#2\Dataset#2\gt",
            batch_size=batch_size,
            num_workers=num_workers,
            )

        case "occ24":
            datamodule = occ24EllipseDataModule(
            r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O24\gt",
            r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O24\images",
            batch_size=batch_size,
            num_workers=num_workers,
            )

        case "occ20":
            datamodule = occ20EllipseDataModule(
            r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O20\images",
            r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O20\gt",
            batch_size=batch_size,
            num_workers=num_workers,
            )

        case "occ16":
            datamodule = occ16EllipseDataModule(
            r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O16\images",
            r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O16\gt",
            batch_size=batch_size,
            num_workers=num_workers,
            )

        case "occ12":
            datamodule = occ12EllipseDataModule(
            r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O12\images",
            r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O12\gt",
            batch_size=batch_size,
            num_workers=num_workers,
            )

        case "occ8":
            datamodule = occ8EllipseDataModule(
            r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O8\images",
            r"D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O8\gt",
            batch_size=batch_size,
            num_workers=num_workers,
            )

        case "occ4":
            datamodule = occ4EllipseDataModule(
            r"D:\Exercises\HIWI\EllipDet-master\Final_dataset\images",
            r"D:\Exercises\HIWI\EllipDet-master\Final_dataset\gt",
            batch_size=batch_size,
            num_workers=num_workers,
            )
        case _:
            raise ValueError(f"Dataset {dataset} not found.")

    if iterations > 1 and (lr is not None or weight_decay is not None):
        print(
            "Warning: Running with multiple iterations with a fixed learning rate or weight decay."
        )
    
    print(f"num-workers: {num_workers}, batch-size: {batch_size}")
    for iteration in range(iterations):
        sampled_lr = random.uniform(lr_min, lr_max)
        sampled_weight_decay = random.uniform(weight_decay_min, weight_decay_max)
        lr = lr if lr is not None else sampled_lr
        weight_decay = (
            weight_decay if weight_decay is not None else sampled_weight_decay
        )

        print(f"Using parameters - Learning rate: {lr}, Weight decay: {weight_decay}")
        print(f"Starting iteration {iteration + 1}/{iterations}")
        pl_module = EllipseRCNNModule(lr=lr, weight_decay=weight_decay)

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss_total",
            dirpath="checkpoints",
            filename=r"loss={val/loss_total:.5f}-e={epoch:02d}",
            auto_insert_metric_name=True,
            save_top_k=-1,
            mode="min",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val/loss_total",
            patience=5,
            mode="min",
        )
        trainer = pl.Trainer(
            accelerator=accelerator,
            precision="32-true",
            max_epochs=40,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

        trainer.fit(pl_module, datamodule=datamodule)
        print(f"Completed iteration {iteration + 1}/{iterations}")


if __name__ == "__main__":
    app()
