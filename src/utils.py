import numpy as np
from PIL import Image
from src.config import IMG_HEIGHT, IMG_WIDTH

def preprocess_image(uploaded_file):
    """
    Standard preprocessing.
    """
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Use high-quality LANCZOS resizing
        img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return image, img_array
    except Exception:
        return None, None