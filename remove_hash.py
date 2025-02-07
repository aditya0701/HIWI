import os

def remove_hash_from_filenames(folder_path):
    # Get a list of all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the filename contains '#'
        if '#' in filename:
            # Create the new filename by removing all '#' characters
            new_filename = filename.replace('#', '')
            original_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            
            # Ensure that the new filename does not already exist to avoid overwriting files
            if os.path.exists(new_file):
                print(f"Cannot rename '{filename}' to '{new_filename}' because '{new_filename}' already exists.")
            else:
                # Rename the file
                os.rename(original_file, new_file)
                print(f"Renamed '{filename}' to '{new_filename}'.")

if __name__ == "__main__":
    # Prompt the user for the folder path
    folder = input("Enter the path to the folder containing the files: ")
    remove_hash_from_filenames(folder)