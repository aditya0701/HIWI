import os
import shutil

def copy_files_to_new_folder(source_folders, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    for source_folder in source_folders:
        # Check if the source folder exists
        if not os.path.exists(source_folder):
            print(f"Source folder does not exist: {source_folder}")
            continue
        
        # Iterate through all files in the source folder
        for filename in os.listdir(source_folder):
            file_path = os.path.join(source_folder, filename)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                try:
                    # Copy the file to the destination folder
                    shutil.copy(file_path, destination_folder)
                    print(f"Copied: {file_path} to {destination_folder}")
                except Exception as e:
                    print(f"Error copying {file_path}: {e}")

# Example usage
source_folders = [
    r'D:\Exercises\HIWI\EllipDet-master\Prasad\Prasad\images',  # Change these paths to your actual folders
    r'D:\Exercises\HIWI\EllipDet-master\Industrial\images',
    r'D:\Exercises\HIWI\EllipDet-master\Dataset#2\Dataset#2\images',
    r'D:\Exercises\HIWI\EllipDet-master\Calibration\Calibration\images'
]
destination_folder = r'D:\Exercises\HIWI\EllipDet-master\Final_dataset\images'  # Destination path

copy_files_to_new_folder(source_folders, destination_folder)