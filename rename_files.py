import os

def rename_files_in_folders(folders):
    for folder in folders:
        # Check if the folder exists
        if not os.path.exists(folder):
            print(f"Folder does not exist: {folder}")
            continue
        
        # Walk through each directory and file in the folder
        for dirpath, _, filenames in os.walk(folder):
            # Extracting the desired folder name (e.g., O16)
            folder_parts = dirpath.split(os.sep)  # Split path into parts
            if len(folder_parts) >= 2:  # Ensure there are enough parts
                final_folder_name = folder_parts[-2]  # Get second last part (e.g., O16)

                for filename in filenames:
                    # Create new filename by appending final folder name before the file extension
                    name, ext = os.path.splitext(filename)
                    new_filename = f"{name}_{final_folder_name}{ext}"
                    
                    # Construct full file paths
                    old_file_path = os.path.join(dirpath, filename)
                    new_file_path = os.path.join(dirpath, new_filename)

                    try:
                        # Rename the file
                        os.rename(old_file_path, new_file_path)
                        print(f"Renamed: {old_file_path} to {new_file_path}")
                    except Exception as e:
                        print(f"Error renaming {old_file_path}: {e}")

# Example usage
folders = [  # Change these paths to your actual folders
    r'D:\Exercises\HIWI\EllipDet-master\Dataset2\Dataset2\gt',
]

rename_files_in_folders(folders)

def rename_files_in_folder_2(folder_path):
    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file name contains '.jpg' or '.bmp'
        if '.jpg' in filename or '.bmp' in filename:
            # Create new file name by removing the extension
            new_filename = filename.replace('.jpg', '').replace('.bmp', '')
            # Construct full file paths
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Rename the file only if the new name is different
            if old_file_path != new_file_path:
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} to {new_file_path}')

# Specify your folder path here
folder_name = r'D:\Exercises\HIWI\EllipDet-master\Dataset2\Dataset2\gt'
rename_files_in_folder_2(folder_name)