import cv2
import numpy as np

def preprocess_image(image_path, img_size=(128, 128)):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize image to fixed size
    image = cv2.resize(image, img_size)
    # Normalize pixel values
    image = image / 255.0
    # Expand dimensions to fit model input
    image = np.expand_dims(image, axis=-1)

    return image
