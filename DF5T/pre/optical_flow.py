import cv2
import numpy as np
import os

def estimate_membrane_flow(image_list, membrane_mask_list, num_frames=3,
                         existence_threshold=0.3, pyr_scale=0.5, levels=3):
    membrane_masks = [mask.astype(np.float32)/255.0 for mask in membrane_mask_list]
    
    refined_masks = []
    
    for idx in range(len(image_list)):
        current_frame = image_list[idx]
        current_mask = membrane_masks[idx]

        accumulated_mask = current_mask.copy()
        valid_count = 1 
        
        for offset in (-1, 1): 
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(image_list):

                flow = cv2.calcOpticalFlowFarneback(
                    prev=image_list[neighbor_idx],
                    next=current_frame,
                    flow=None,
                    pyr_scale=pyr_scale,
                    levels=levels,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
                

                h, w = current_mask.shape
                map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                map_xy = (np.dstack((map_x, map_y)) + flow).astype(np.float32)
                

                remapped_mask = cv2.remap(
                    membrane_masks[neighbor_idx],
                    map_xy[...,0],
                    map_xy[...,1],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
                valid_region = (remapped_mask > 0.1).astype(np.float32)
                accumulated_mask += remapped_mask * valid_region
                valid_count += np.sum(valid_region)
        

        avg_mask = accumulated_mask / valid_count
        _, binary_mask = cv2.threshold((avg_mask*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        refined_masks.append(refined)
    return refined_masks

def process_with_refined_mask(enhanced_images, refined_masks, output_folder):
    final_images = []
    
    for i, (img, mask) in enumerate(zip(enhanced_images, refined_masks)):
        try:
            mask_float = mask.astype(np.float32) / 255.0 
            enhanced = np.where(
                mask_float > 0, 
                cv2.addWeighted(img, 0.9, (mask_float * 255).astype(np.uint8), 0.3, 0),
                img
            )
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            final_images.append(enhanced)
            
            result_path = os.path.join(output_folder, f"enhanced_complete_{i+1:04d}.png")
            cv2.imwrite(result_path, enhanced)
        
        except Exception as e:
            continue
    
    return final_images

