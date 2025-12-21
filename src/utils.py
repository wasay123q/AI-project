import numpy as np
from PIL import Image
from src.config import IMG_HEIGHT, IMG_WIDTH

def preprocess_image(uploaded_file):
    """
    Loads, resizes, and normalizes an image for the model.
    """
    try:
        # 1. Open the image
        image = Image.open(uploaded_file).convert('RGB')
        
        # 2. Resize to 128x128 (Crucial for the new model)
        # Uses High-Quality resampling
        img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        
        # 3. Convert to Array & Normalize
        img_array = np.array(img_resized) / 255.0
        
        # 4. Expand dimensions to (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return image, img_array
        
    except Exception as e:
        return None, None