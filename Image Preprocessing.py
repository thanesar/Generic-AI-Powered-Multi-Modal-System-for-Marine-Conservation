import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """Loads an image from the specified path in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def resize_image(image, width=None, height=None):
    """Resizes the image to the specified width and height while maintaining aspect ratio."""
    if width is None and height is None:
        return image  # No resizing needed
    (h, w) = image.shape[:2]
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def reduce_noise(image):
    """Applies Gaussian Blur to reduce noise in the image."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def save_image(output_path, image):
    """Saves the processed image to the specified path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Processed image saved at: {output_path}")

def display_images(original, processed, title_original="Original Image", title_processed="Processed Image"):
    """Displays the original and processed images side by side."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(title_original)
    plt.imshow(original, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title(title_processed)
    plt.imshow(processed, cmap='gray')
    
    plt.show()

def preprocess_image(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8), resize_dims=(500, 500)):
    """Full pipeline for loading, resizing, noise reduction, CLAHE, and saving an image."""
    try:
        # Load the image
        image = load_image(image_path)
        
        # Resize the image
        resized_image = resize_image(image, width=resize_dims[0], height=resize_dims[1])
        
        # Reduce noise
        noise_reduced_image = reduce_noise(resized_image)
        
        # Apply CLAHE
        clahe_image = apply_clahe(noise_reduced_image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        
        # Save the processed image
        save_image(output_path, clahe_image)
        
        # Display the images
        display_images(original=image, processed=clahe_image)
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Parameters for processing
image_path = "your_image_path.jpg"  # Path to your input image
output_path = "output/processed_image.jpg"  # Path to save the processed image
clip_limit = 2.5  # Adjust for different levels of contrast
tile_grid_size = (8, 8)  # Grid size for CLAHE
resize_dims = (500, 500)  # Resize dimensions (width, height)

# Run the preprocessing pipeline
preprocess_image(image_path, output_path, clip_limit, tile_grid_size, resize_dims)
