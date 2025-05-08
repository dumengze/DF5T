import os
import mrcfile
import cv2
import argparse
import numpy as np
import warnings
from pre.enhance import process_images_in_folder
from optical_flow import estimate_membrane_flow

warnings.filterwarnings('ignore', category=RuntimeWarning, module='mrcfile')

def save_mrc_slices_as_images(mrc_path, output_folder):
    try:
        os.makedirs(output_folder, exist_ok=True)
        image_list = []
        
        print(f"Processing MRC file: {mrc_path}")
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data
            
            if data is None or data.size == 0:
                print(f"Warning: The MRC file {mrc_path} contains empty data.")
                return []
                
            if data.ndim == 3:
                num_slices = data.shape[0]
            else:
                num_slices = 1
                data = data[np.newaxis,...]
            
            print(f"Processing {num_slices} slices...")
            for i in range(num_slices):
                try:
                    slice_data = data[i].copy()
                
                    if np.all(slice_data == 0):
                        print(f"Warning: Slice {i+1} contains all zero values.")
                        continue

                    slice_data = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    # slice_data = cv2.resize(slice_data, (256, 256), interpolation=cv2.INTER_LINEAR)
                    
                    output_path = os.path.join(output_folder, f'slice_{i+1:04d}.png')
                    cv2.imwrite(output_path, slice_data)
                    
                    if os.path.exists(output_path):
                        image_list.append(output_path)
                        print(f"Slice {i+1}/{num_slices} saved successfully.")
                    else:
                        print(f"Warning: Failed to save slice {i+1}.")
                        
                except Exception as e:
                    print(f"Error processing slice {i+1}: {e}")
                    continue
                    
        print(f"Successfully saved {len(image_list)} slices.")
        return image_list
        
    except Exception as e:
        print(f"Successfully saved {len(image_list)} slices.")
        return []

def split_and_process_mrc_images(mrc_path, output_folder, **kwargs):
    image_list = save_mrc_slices_as_images(mrc_path, output_folder)
    if not image_list:
        return [], [], []
    enhanced_images, membrane_mask_list = process_images_in_folder(output_folder, **kwargs)
    return enhanced_images, membrane_mask_list, image_list

def main():
    prog = argparse.ArgumentParser(description="Process MRC files and enhance membrane structures using optical flow prediction.")
    prog.add_argument('--input_folder', type=str, default='./data', help='Path to the folder containing MRC files.')
    prog.add_argument('--output_base_folder', type=str, default='./out', help='Base folder for saving processed images.')
    prog.add_argument('--membrane_gray_min', type=float, default=1, help='Minimum gray value for membrane processing.')
    prog.add_argument('--top_percent', type=float, default=70, help='Maximum gray value for membrane processing.')
    prog.add_argument('--density_threshold', type=float, default=0.52, help='Density threshold for noise removal.')
    prog.add_argument('--dispersion_ratio', type=float, default=0.52, help='Dispersion ratio for processing.')
    prog.add_argument('--denoise_strength', type=float, default=0.0000000001, help='Strength of the denoising filter.')
    prog.add_argument('--color_enhance_factor', type=float, default=0.005, help='Compression factor for processing.')
    prog.add_argument('--noise_compression_factor', type=float, default=0.2, help='Compression factor for processing.')
    prog.add_argument('--window_size', type=float, default=7, help='Denosing window size in mask.')
    prog.add_argument('--num_frames', type=int, default=2, help='Number of frames to consider for optical flow.')
    prog.add_argument('--existence_threshold', type=int, default=0.3, help='Threshold for noise existence across frames.')
    args = prog.parse_args()

    os.makedirs(args.output_base_folder, exist_ok=True)
    
    for file_name in os.listdir(args.input_folder):
        if not file_name.endswith('.mrc'):
            continue
            
        try:
            print(f"Processing file: {file_name}\n")
            mrc_path = os.path.join(args.input_folder, file_name)
            output_folder = os.path.join(args.output_base_folder, os.path.splitext(file_name)[0])
            original_slices = save_mrc_slices_as_images(mrc_path, output_folder)
            if not original_slices:
                continue
            print("Performing mitochondrial membrane enhancement...")
            enhanced_images, base_masks = process_images_in_folder(
                output_folder,
                membrane_gray_min=args.membrane_gray_min,
                top_percent=args.top_percent,
                density_threshold=args.density_threshold,
                dispersion_ratio=args.dispersion_ratio,
                denoise_strength=args.denoise_strength,
                color_enhance_factor=args.color_enhance_factor,
                noise_compression_factor=args.noise_compression_factor,
                window_size=args.window_size
            )
            print("Performing optical flow completion processing...")

            refined_masks = estimate_membrane_flow(
                image_list=enhanced_images,  
                membrane_mask_list=base_masks,
                num_frames=args.num_frames,
                existence_threshold=args.existence_threshold
            )

            final_images = []
            for idx, (orig_img, refined_mask) in enumerate(zip(enhanced_images, refined_masks)):
                try:
                    mask_float = cv2.GaussianBlur(refined_mask.astype(np.float32), (5,5), 0) / 255.0                    
                    darkened = orig_img.astype(np.float32) * (1 - 0.3 * mask_float)                     
                    final_img = np.where(
                        mask_float > 0.1,
                        darkened,
                        orig_img.astype(np.float32)
                    )

                    final_img = np.clip(final_img, 0, 255).astype(np.uint8)

                    result_path = os.path.join(output_folder, f"final_{idx+1:04d}.png")
                    cv2.imwrite(result_path, final_img)
                    final_images.append(final_img)
                    print(f"Final image saved: {result_path}")
                    
                except Exception as e:
                    print(f"Image composition failed {idx+1}: {e}")
                    continue
                    
            print(f"Processing completed for file: {file_name}\n")
            
        except Exception as e:
            print(f"File processing failed: {file_name}: {e}")

if __name__ == "__main__":
    main()