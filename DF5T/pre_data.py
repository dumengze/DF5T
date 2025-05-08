import os
from natsort import natsorted
def list_images_in_folder(folder_path, output_file):
    valid_extensions = ['.png', '.tif', '.jpg', '.jpeg', '.tiff']
    valid_files = [
        filename for filename in os.listdir(folder_path) 
        if any(filename.lower().endswith(ext) for ext in valid_extensions)
    ]
    sorted_files = natsorted(valid_files) 
    with open(output_file, 'w') as f:
        for filename in sorted_files:
            name_without_extension = os.path.splitext(filename)[0]
            f.write(f"{name_without_extension} 1\n")
folder_path = "exp/datasets/MitEM/MitEM"  
output_file = "exp/MitEM_val_1k.txt" 
list_images_in_folder(folder_path, output_file)

