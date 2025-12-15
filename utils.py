from PIL import Image
import requests
from io import BytesIO
import os

def show_image_in_notebook(image):
    """Display a PIL Image in a Jupyter notebook."""
    from IPython.display import display
    display(image)

def load_image(image_path, resize=False, max_size=512):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    if resize:
        image = resize_image(image, max_size=max_size)
    return image

def load_image_as_list(image_paths:list, resize=False, max_size=512):
    images = []
    for path in image_paths:
        img = load_image(path, resize=resize, max_size=max_size)
        images.append(img)
    return images


def resize_image(image, max_size=512, maintain_aspect=True):
    """
    Resize image conservatively, maintaining aspect ratio and not exceeding max_size.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension (width or height)
        maintain_aspect: If True, maintains aspect ratio
    
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        # Calculate new size maintaining aspect ratio
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # Only resize if image is larger than max_size
        if width > max_size or height > max_size:
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image.copy()
    else:
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    


from PIL import Image, ImageOps

def optimize_image(image_path, output_path=None, target_size=1120, tile_size=560):
    """
    Resizes and pads an image to optimal dimensions for Llama 4 vision models.
    
    Strategy:
    1. Resizes the longest edge to 'target_size' (1120px) while maintaining aspect ratio.
    2. Pads the shorter edge to align with 'target_size' (resulting in a square 1120x1120 canvas).
    3. Uses high-quality Lanczos resampling to prevent aliasing on text/lines.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str, optional): Path to save the optimized image. If None, returns PIL object.
        target_size (int): The target dimension for the square canvas (default 1120 for 2x2 grid).
        tile_size (int): The underlying tile size (default 560). 
                         (Used here for logic validation, though target_size is the driver).
    
    Returns:
        PIL.Image: The processed image object.
    """
    
    img = load_image(image_path)
    original_width, original_height = img.size
    
    # 1. Calculate new dimensions maintaining aspect ratio
    aspect_ratio = original_width / original_height
    
    if aspect_ratio > 1:
        # Landscape: Width becomes target_size
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        # Portrait/Square: Height becomes target_size
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
        
    # 2. Resize using high-quality Lanczos resampling (best for downscaling)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 3. Pad to create a square canvas (1120x1120)
    # We use a neutral color (black or grey) for padding. 
    # Llama 4 generally handles black padding well.
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    
    padding = (
        delta_w // 2,        # Left
        delta_h // 2,        # Top
        delta_w - (delta_w // 2), # Right
        delta_h - (delta_h // 2)  # Bottom
    )
    
    # Add padding (fill with black: (0,0,0))
    img_padded = ImageOps.expand(img_resized, padding, fill=(0, 0, 0))
    
    # 4. Save or Return
    if output_path:
        img_padded.save(output_path, quality=95)
        print(f"Image saved to {output_path} | Size: {img_padded.size}")
    
    return img_padded

# --- Usage Example ---
# if __name__ == "__main__":
#     # Example usage for a wide chart
#     processed_img = optimize_image_for_llama4("input_chart.png", "ready_for_llama.jpg")
