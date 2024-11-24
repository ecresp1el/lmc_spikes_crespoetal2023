import os
import shutil

# Define the base directories
base_dirs = [
    "/Volumes/Manny2TB/mea_blade_round3_led_ctz",
    "/Volumes/Manny2TB/mea_blade_round4_led_ctz",
    "/Volumes/Manny2TB/mea_blade_round5_led_ctz",
]

# Subdirectory containing the .h5 files
h5_subdir = "h5_files"

# Iterate over each base directory
for base_dir in base_dirs:
    h5_dir = os.path.join(base_dir, h5_subdir)
    if os.path.exists(h5_dir):
        # Loop through all files in the h5_files directory
        for file_name in os.listdir(h5_dir):
            if file_name.endswith(".h5"):
                # Extract base name without timestamp or extension
                base_name = file_name.split(".")[0]
                
                # Create the new directory for the file
                new_dir = os.path.join(base_dir, base_name)
                os.makedirs(new_dir, exist_ok=True)
                
                # Copy the file into the new directory
                source_file = os.path.join(h5_dir, file_name)
                destination_file = os.path.join(new_dir, file_name)
                shutil.copy2(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")
    else:
        print(f"Directory {h5_dir} does not exist.")