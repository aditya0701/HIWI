import requests
import zipfile
import h5py
import os
from pathlib import Path
from ellipse_rcnn.data.base import EllipseDatasetBase

class GitHubDataset(EllipseDatasetBase):
    def __init__(self, root: Path, download: bool = False) -> None:
        self.root = root
        self.zip_url = "https://github.com/zikai1/EllipseDetection/blob/master/Industrial.zip"
        self.zip_file = self.root / "dataset.zip"
        self.extracted_dir = self.root / "repo-main"
        self.hdf5_file = self.root / "dataset.h5"

        if download:
            self.download_and_process()

    def download_and_process(self) -> None:
        # Create root directory
        self.root.mkdir(parents=True, exist_ok=True)

        # Download ZIP file
        print(f"Downloading dataset from {self.zip_url}...")
        response = requests.get(self.zip_url)
        with open(self.zip_file, "wb") as f:
            f.write(response.content)

        # Extract ZIP file
        print("Extracting dataset...")
        with zipfile.ZipFile(self.zip_file, "r") as zip_ref:
            zip_ref.extractall(self.root)

        # Process extracted data and save to HDF5
        print("Processing dataset to HDF5...")
        self.process_to_hdf5()

    def process_to_hdf5(self) -> None:
        # Example: Writing extracted data to HDF5
        with h5py.File(self.hdf5_file, "w") as hf:
            # Iterate through extracted dataset (modify as needed)
            for i, file in enumerate(os.listdir(self.extracted_dir)):
                if file.endswith(".jpg"):
                    # Example: Storing dummy data
                    hf.create_dataset(f"image_{i}", data=[1, 2, 3])  # Replace with actual image data processing

        print(f"Dataset saved to {self.hdf5_file}")
