import cv2
import numpy as np
import os

# Function to enhance contrast using CLAHE
def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    return clahe.apply(image)

import numpy as np
import cv2

def preprocess_image(image, membrane_gray_min=50, top_percent=10):
    # Enhance contrast first
    gray = enhance_contrast(image)

    # Get pixels and sort them by gray value
    pixels = gray.flatten()
    sorted_pixels = np.sort(pixels)
    
    # Calculate membrane gray max using top_percent
    membrane_gray_max = np.percentile(sorted_pixels, 100 - top_percent)

    # Create a mask based on gray values for membrane extraction
    membrane_mask = cv2.inRange(gray, membrane_gray_min, membrane_gray_max)
    print(f"Membrane max gray value: {membrane_gray_max}")
    
    # Extract membrane regions based on the mask
    membrane_gray = gray * (membrane_mask > 0)

    return gray, membrane_mask, membrane_gray



# Function to detect membrane edges with sharp edge detection
def find_membranes_edges(membrane_mask):
    # Use Canny edge detection with sharper settings for clear membrane edges
    edges = cv2.Canny(membrane_mask, 50, 80)
    return edges

def enhance_membrane(gray, membrane_mask, noise_reduction_level_1=70, noise_enhance_level_2_3=70):
    # Extract membrane region
    membrane_pixels = gray[membrane_mask > 0]
    
    # Define intensity thresholds based on percentiles
    light_threshold = np.percentile(membrane_pixels, 90)
    dark_threshold = np.percentile(membrane_pixels, 10)
    
    # Classify pixels into three categories
    light_pixels = gray > light_threshold
    mid_dark_pixels = (gray >= dark_threshold) & (gray <= light_threshold)
    dark_pixels = gray < dark_threshold

    enhanced_image = gray.copy()

    # Light pixel noise reduction (20%)
    enhanced_image[light_pixels] = enhanced_image[light_pixels] - (enhanced_image[light_pixels] * noise_reduction_level_1 / 100)
    
    # Dark and mid-dark pixel noise enhancement (enhance dark parts)
    enhanced_image[mid_dark_pixels] = enhanced_image[mid_dark_pixels] + (255 - enhanced_image[mid_dark_pixels]) * noise_enhance_level_2_3 / 100
    enhanced_image[dark_pixels] = enhanced_image[dark_pixels] + (255 - enhanced_image[dark_pixels]) * noise_enhance_level_2_3 / 100
    
    return enhanced_image

def lighten_and_denoise(image, mitochondria_mask, denoise_strength=0.1):
    """
    Function to lighten the background and apply non-local means denoising.
    denoise_strength: float from 0 to 1. 
    """
    # Check denoise_strength is within valid range
    if denoise_strength < 0.0 or denoise_strength > 1.0:
        print(f"Warning: denoise_strength should be between 0 and 1. Using default value of 0.1.")
        denoise_strength = 0.1
    
    # If denoise_strength is 0, return the original image without any processing
    if denoise_strength == 0:
        return image
    
    # Create background mask
    background_mask = cv2.bitwise_not(mitochondria_mask)
    
    # Extract background area from original image
    background = cv2.bitwise_and(image, image, mask=background_mask)
    
    # Apply non-local means denoising to background (for original image)
    denoised_background = cv2.fastNlMeansDenoising(background, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Lighten the denoised background
    lightened_background = denoised_background * (1 - denoise_strength) + 255 * denoise_strength
    
    # Combine lightened background with the regions that are part of the membrane
    lightened_image = image.copy()
    lightened_image[background_mask > 0] = lightened_background[background_mask > 0]
    
    return lightened_image




def process_mitochondria(image, mitochondria_mask, color_enhance_factor=0.5, noise_compression_factor=0.5, repair_gap_factor=0.5):
    """
    Enhance the mitochondria mask by adjusting intensity (darkening), refining the mask shape, and reconstructing broken mitochondrial membranes.
    """
    # Ensure the mask is binary (0 or 255)
    mitochondria_mask = (mitochondria_mask > 0).astype(np.uint8) * 255
    avg_gray = np.average(mitochondria_mask)
    print(f"Average Gray Value: {avg_gray}")
    
    # Adjust color_enhance_factor based on average gray value
    if avg_gray < 15:
        color_enhance_factor = 0.15
    elif 15 <= avg_gray < 60:
        color_enhance_factor = 0.12
    elif 60 <= avg_gray < 125:
        color_enhance_factor = 0.09
    elif 125 <= avg_gray < 180:
        color_enhance_factor = 0.06
    else:  # avg_gray >= 180
        color_enhance_factor = 0.03
    print(f"Using color_enhance_factor: {color_enhance_factor}")

    # Apply non-linear darkening to mitochondria mask (more darkening when color_enhance_factor is higher)
    enhanced_mask = mitochondria_mask.copy()

    # Darkening effect: Reduce pixel intensity towards 0 based on color_enhance_factor
    enhanced_mask[enhanced_mask > 0] = np.clip(
        enhanced_mask[enhanced_mask > 0] - (enhanced_mask[enhanced_mask > 0] * color_enhance_factor),  # Darken the pixels
        1, 254  # Ensure values stay within valid mask range
    )
    
    # Enhance image based on refined mask (darken intensity in the mitochondria regions)
    enhanced_image = image.copy()
    
    # Apply darkening to the mitochondria regions
    enhanced_image[enhanced_mask > 0] = np.clip(enhanced_image[enhanced_mask > 0] - (enhanced_image[enhanced_mask > 0] * color_enhance_factor), 1, 254)
    
    # Ensure the enhanced image is properly adjusted (no "transparent" areas)
    enhanced_image = np.uint8(enhanced_image)  # Ensure proper data type for the image

    # Return both enhanced image and the refined mitochondria mask (now darkened)
    return enhanced_image, enhanced_mask





def detect_membrane_regions_with_dense_noise(image, membrane_mask, membrane_gray, window_size=4, density_threshold=0.5, dilation_iterations=2, erosion_iterations=2, min_cluster_size_ratio=0.02):
    height, width = membrane_mask.shape
    dense_mask = np.zeros_like(membrane_mask)
    dense_mask_before_morph = np.zeros_like(membrane_mask)
    
    # Convert membrane_mask to binary (0s and 1s)
    membrane_mask_binary = (membrane_mask > 0).astype(int)
    
    noise_points = []
    window_area = (2 * window_size + 1) ** 2
    
    for y in range(height):
        for x in range(width):
            if membrane_mask_binary[y, x] > 0:  # Membrane pixel
                # Define window boundaries
                y_min = max(0, y - window_size)
                y_max = min(height, y + window_size + 1)
                x_min = max(0, x - window_size)
                x_max = min(width, x + window_size + 1)
                
                # Extract local membrane mask binary
                local_window = membrane_mask_binary[y_min:y_max, x_min:x_max]
                local_density = np.sum(local_window)
                density_ratio = local_density / window_area
                
                # Use membrane_gray to refine noise detection
                local_gray_value = membrane_gray[y, x]
                # Adjust dynamic threshold to subtract a fraction of local gray value
                dynamic_density_threshold = density_threshold - (local_gray_value / 255.0) * 0.1
                # Ensure dynamic threshold does not go below a minimum value
                dynamic_density_threshold = max(dynamic_density_threshold, 0.3)
                
                if density_ratio > dynamic_density_threshold and local_gray_value > 0:
                    noise_points.append((y, x))
    
    # Mark dense noise points
    for (y, x) in noise_points:
        dense_mask_before_morph[y, x] = 255
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    dense_mask = cv2.dilate(dense_mask_before_morph, kernel, iterations=dilation_iterations)
    dense_mask = cv2.erode(dense_mask, kernel, iterations=erosion_iterations)
    
    # Remove small clusters
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dense_mask, connectivity=8)
    min_cluster_size = int(height * width * min_cluster_size_ratio)
    
    for i in range(1, num_labels):
        cluster_size = stats[i, cv2.CC_STAT_AREA]
        if cluster_size < min_cluster_size:
            dense_mask[labels == i] = 0
    
    # Filter the original membrane mask to keep only the dense regions
    final_mask = cv2.bitwise_and(membrane_mask, dense_mask)
    
    # Save intermediate masks for visualization
    return final_mask



def process_and_color_membrane(image_path, membrane_gray_min=50, top_percent=10, density_threshold=0.35, dispersion_ratio=0.1, denoise_strength=0.1, color_enhance_factor=0.2, noise_compression_factor=0.2, window_size=10):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Skipping.")
        return
    
    gray, membrane_mask, membrane_gray = preprocess_image(image, membrane_gray_min, top_percent)
    
    # cv2.imwrite(f"{os.path.splitext(image_path)[0]}_top_percent_{top_percent}_membrane_mask.png", membrane_mask)

    dense_mask = detect_and_color_dense_noise_points(
        gray, membrane_mask, window_size=window_size, dispersion_ratio=dispersion_ratio
    )

    # cv2.imwrite(f"{os.path.splitext(image_path)[0]}_dispersion_ratio_{dispersion_ratio}_dense_mask.png", dense_mask)

    lightened_image = lighten_and_denoise(image, dense_mask, denoise_strength=denoise_strength)

    dense_region = cv2.bitwise_and(image, image, mask=dense_mask)

    enhanced_image, refined_mask = process_mitochondria(dense_region, dense_mask, color_enhance_factor=color_enhance_factor, noise_compression_factor=noise_compression_factor)
    
    # Here we ensure that black regions in the refined_mask are removed before combining it with lightened_image
    refined_mask_non_black = np.where(refined_mask > 0, refined_mask, 0)  # Remove black regions (background)

    # To ensure we don't get black backgrounds, we combine images based on the refined mask
    # We use the refined mask as a weight, ensuring smooth blending
    refined_mask_non_black_float = refined_mask_non_black.astype(float) / 255  # Normalize mask to [0, 1]
    
    # Final image blending using the refined mask (after removal of black areas)
    final_image = (lightened_image.astype(float) * (1 - refined_mask_non_black_float) + enhanced_image.astype(float) * refined_mask_non_black_float).astype(np.uint8)

    # Save the final processed image
    # cv2.imwrite(f"{os.path.splitext(image_path)[0]}_processed.png", final_image)

    return final_image, refined_mask_non_black




def process_images_in_folder(folder_path, membrane_gray_min=1, top_percent=10, density_threshold=0.35, dispersion_ratio=0.1, denoise_strength=0.4, color_enhance_factor=0.2, noise_compression_factor=0.2, window_size=10):
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return [], []
    enhanced_images = []
    membrane_masks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            image_path = os.path.join(folder_path, filename)
            enhanced_img, refined_mask = process_and_color_membrane(
                image_path,
                membrane_gray_min,
                top_percent,
                density_threshold,
                dispersion_ratio,
                denoise_strength,
                color_enhance_factor,
                noise_compression_factor,
                window_size
            )
            if enhanced_img is not None and refined_mask is not None:
                enhanced_images.append(enhanced_img)
                membrane_masks.append(refined_mask)
    return enhanced_images, membrane_masks

def detect_and_color_dense_noise_points(image, membrane_mask, window_size=30, dispersion_ratio=0.1, noise_compression_factor=0.3):
    height, width = membrane_mask.shape
    dense_mask = np.zeros_like(membrane_mask)

    window_area = (2 * window_size + 1) ** 2
    
    for y in range(height):
        for x in range(width):
            if membrane_mask[y, x] > 0:  
                y_min = max(0, y - window_size)
                y_max = min(height, y + window_size + 1)
                x_min = max(0, x - window_size)
                x_max = min(width, x + window_size + 1)

                window = membrane_mask[y_min:y_max, x_min:x_max]
                mask_pixels_in_window = np.sum(window > 0) 
                density = mask_pixels_in_window / window_area

                if density >= dispersion_ratio:
                    dense_mask[y, x] = membrane_mask[y, x]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  
    dense_mask = cv2.morphologyEx(dense_mask, cv2.MORPH_OPEN, kernel)  


    dense_mask[membrane_mask == 0] = 0 

    if noise_compression_factor > 0:

        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  
        dense_mask = cv2.dilate(dense_mask, dilation_kernel, iterations=int(noise_compression_factor * 2)) 
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        dense_mask = cv2.erode(dense_mask, erosion_kernel, iterations=int(noise_compression_factor * 2))

    return dense_mask

# if __name__ == "__main__":
#     folder_path = r"dmz/dataset"
#     process_images_in_folder(
#         folder_path,
#         membrane_gray_min=1,
#         top_percent=70,
#         density_threshold=0.001,
#         dispersion_ratio=0.001,
#         denoise_strength=0.3,
#         color_enhance_factor=0.25,
#         noise_compression_factor=0.2,
#         window_size=1 

if __name__ == "__main__":
    folder_path = "dmz/dataset"
    process_images_in_folder(
        folder_path,
        membrane_gray_min=1,
        top_percent=80,
        density_threshold=0.7,
        dispersion_ratio=0.7,
        denoise_strength=0,
        color_enhance_factor=0.005,
        noise_compression_factor=0.2,
        window_size=3 
    )
    
    