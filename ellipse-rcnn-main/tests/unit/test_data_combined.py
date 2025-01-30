import random
from typing import List, Tuple, Optional

from torch.utils.data import Dataset, ConcatDataset, DataLoader


class MockEllipseDataset(Dataset):
    """
    A mock dataset that simulates image and annotation files.
    """
    def __init__(
        self,
        image_files: List[str],
        annotation_files: List[str],
        transform: Optional = None,
        resize: Tuple[int, int] = (640, 640),
    ):
        assert len(image_files) == len(annotation_files), "Number of images and annotations must match."
        self.image_files = image_files
        self.annotation_files = annotation_files
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # For simplicity, return a tuple of image and annotation file names
        image = self.image_files[idx]
        annotation = self.annotation_files[idx]
        if self.transform:
            # Apply transformations if any (not used in this mock)
            image = self.transform(image)
        return image, annotation

# 2. Define the CombinedEllipseDataModule
class CombinedEllipseDataModule:
    """
    Combines multiple ellipse datasets and splits them into train, val, and test sets.
    """
    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int,
        num_workers: int,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
        resize: Tuple[int, int] = (640, 640),
        transform: Optional = None,
    ):
        assert train_split + val_split + test_split == 1.0, "Splits must sum to 1."
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.resize = resize
        self.transform = transform

    def setup(self):
        """
        Splits each dataset into train, val, and test subsets and combines them.
        """
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

        for dataset in self.datasets:
            total = len(dataset)
            indices = list(range(total))
            random.seed(self.seed)
            random.shuffle(indices)

            train_end = int(self.train_split * total)
            val_end = train_end + int(self.val_split * total)

            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            test_subset = torch.utils.data.Subset(dataset, test_indices)

            self.train_datasets.append(train_subset)
            self.val_datasets.append(val_subset)
            self.test_datasets.append(test_subset)

        # Combine all train, val, and test datasets
        self.train_dataset = ConcatDataset(self.train_datasets) if self.train_datasets else None
        self.val_dataset = ConcatDataset(self.val_datasets) if self.val_datasets else None
        self.test_dataset = ConcatDataset(self.test_datasets) if self.test_datasets else None

    def get_dataloader(self, dataset: ConcatDataset, shuffle: bool = False):
        """
        Returns a DataLoader for the given dataset.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        ) if dataset else None

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function if needed. Here, it's the default.
        """
        return batch

    def print_summary(self):
        """
        Prints the number of samples in each split and some example entries.
        """
        print("===== Combined Ellipse Data Module Summary =====")
        if self.train_dataset:
            print(f"Training Samples: {len(self.train_dataset)}")
            # Print first 3 samples
            print("First 3 Training Samples:")
            for i in range(min(3, len(self.train_dataset))):
                print(f"  Sample {i+1}: {self.train_dataset[i]}")
        else:
            print("No Training Dataset.")

        if self.val_dataset:
            print(f"\nValidation Samples: {len(self.val_dataset)}")
            # Print first 3 samples
            print("First 3 Validation Samples:")
            for i in range(min(3, len(self.val_dataset))):
                print(f"  Sample {i+1}: {self.val_dataset[i]}")
        else:
            print("No Validation Dataset.")

        if self.test_dataset:
            print(f"\nTesting Samples: {len(self.test_dataset)}")
            # Print first 3 samples
            print("First 3 Testing Samples:")
            for i in range(min(3, len(self.test_dataset))):
                print(f"  Sample {i+1}: {self.test_dataset[i]}")
        else:
            print("No Testing Dataset.")

        print("===============================================")

# 3. Instantiate and Setup the Data Module with Mock Data
def main():
    # Create mock data for two datasets
    # Dataset 1: Industry
    industry_images = [f"industry_image_{i}.jpg" for i in range(10)]
    industry_annotations = [f"industry_gt_{i}.txt" for i in range(10)]
    industry_dataset = MockEllipseDataset(
        image_files=industry_images,
        annotation_files=industry_annotations,
    )

    # Dataset 2: Prasad
    prasad_images = [f"prasad_image_{i}.png" for i in range(6)]
    prasad_annotations = [f"prasad_gt_{i}.txt" for i in range(6)]
    prasad_dataset = MockEllipseDataset(
        image_files=prasad_images,
        annotation_files=prasad_annotations,
    )

    # Combine datasets
    combined_datamodule = CombinedEllipseDataModule(
        datasets=[industry_dataset, prasad_dataset],
        batch_size=4,
        num_workers=0,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        seed=42,
    )

    # Setup the data module (split datasets)
    combined_datamodule.setup()

    # Print summary for manual verification
    combined_datamodule.print_summary()

    # Optionally, iterate through a DataLoader
    train_loader = combined_datamodule.get_dataloader(combined_datamodule.train_dataset, shuffle=True)
    print("\nIterating through Training DataLoader:")
    for batch in train_loader:
        print(batch)
        break  # Just show the first batch

if __name__ == "__main__":
    main()