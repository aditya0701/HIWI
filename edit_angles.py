import os
import math

def convert_to_radians(angle):
    """Convert degrees to radians if the angle is in degrees."""
    return angle * (math.pi / 180)

def process_angle_files(directory):
    # Iterate through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                updated_lines = []  # To store updated lines with valid angles
                angles = []  # To collect angles for validation

                # First pass: Collect angles and check their validity
                for index, line in enumerate(lines):
                    params = line.strip().split()
                    if not params:
                        continue
                    
                    # Skip the first line which contains only a number
                    if index == 0:
                        continue
                    
                    if len(params) == 5:  # Ensure there are 5 parameters
                        try:
                            # Extract the angle (assuming it's the last parameter)
                            angle = float(params[-1])
                            angles.append(angle)  # Store original angle for later processing
                        except ValueError:
                            print(f"Invalid number in line: {line.strip()}")
                            updated_lines.append(line.strip())  # Keep original line if error
                            continue
                    
                    else:
                        print(f"Unexpected format in line: {line.strip()}, file: {filename}")
                        updated_lines.append(line.strip())  # Keep original line if error
                
                # Check if any angle is outside the radian range
                needs_conversion = any(angle < -(2 * math.pi) or angle >= 2 * math.pi for angle in angles)

                # Second pass: Process lines again to update them based on conversion need
                for index, line in enumerate(lines):
                    params = line.strip().split()
                    if not params:
                        continue
                    
                    # Skip the first line which contains only a number
                    if index == 0:
                        continue
                    
                    if len(params) == 5:  
                        try:
                            # Extract and possibly convert the angle 
                            angle = float(params[-1])
                            
                            if needs_conversion:
                                converted_angle = convert_to_radians(angle)
                                params[-1] = str(converted_angle)
                            
                            updated_lines.append(' '.join(params))
                        
                        except ValueError:
                            print(f"Invalid number in line: {line.strip()}")
                            updated_lines.append(line.strip())  

                    else:
                        print(f"Unexpected format in line: {line.strip()}, file: {filename}")
                        updated_lines.append(line.strip())  

                # Write back the updated lines to the same file, including skipped first line.
                with open(file_path, 'w') as file:
                    # Write back the first unchanged line (the count)
                    if lines and len(lines) > 0 and lines[0].strip():
                        file.write(lines[0])  
                    
                    for updated_line in updated_lines:
                        file.write(updated_line + '\n')

            except IOError as e:
                print(f"Error processing file {filename}: {e}")

# Usage example
folder_path = r'D:\Exercises\HIWI\EllipDet-master\Prasad\Prasad\gt'  # Change this to your folder path
process_angle_files(folder_path)