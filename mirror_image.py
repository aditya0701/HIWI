import os

def fix_ellipse_params(folder_path, image_height):
    # Loop over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if it's a file (you can add more checks here if needed)
        if os.path.isfile(file_path):
            # Open the file and read the data
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Skip empty files
            if not lines:
                continue

            # First line is the number of ellipses
            num_ellipses = int(lines[0].strip())
            new_lines = [lines[0]]  # Keep the first line as is

            # Process each ellipse
            for line in lines[1:]:
                # Parse the values
                parts = line.strip().split()
                if len(parts) != 5:
                    # Line does not have 5 values, skip or handle accordingly
                    continue

                xc, yc, width, height, angle = map(float, parts)

                # Fix the y-coordinate
                yc_new = image_height - yc

                # Fix the angle
                angle_new = -angle

                # Build the new line
                new_line = f"{xc} {yc_new} {width} {height} {angle_new}\n"
                new_lines.append(new_line)

            # Write the corrected data back to the file
            with open(file_path, 'w') as f:
                f.writelines(new_lines)

if __name__ == "__main__":
    folder = input(r"enter file name")
    image_height = float(input("Enter the image height (in the same units as yc): "))
    fix_ellipse_params(folder, image_height)
# # Example usage
# folder_path = r'D:\Exercises\HIWI\EllipDet-master\Occluded\Occluded\O4\gt'  # Change this to your folder path
# fix_ellipse_parameters(folder_path)