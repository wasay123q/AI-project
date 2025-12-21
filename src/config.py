import os

# SYSTEM SETTINGS
# The name of the file you will download from Colab later
MODEL_NAME = "deepdetect_mobilenet_128.h5"
MODEL_PATH = os.path.join("models", MODEL_NAME)

# IMAGE SETTINGS 
# These must match the training script in Colab EXACTLY
IMG_HEIGHT = 128
IMG_WIDTH = 128
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# CLASS LABELS
CLASS_NAMES = ["AI Generated", "Real"]