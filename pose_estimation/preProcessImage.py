import cv2
import numpy as np

def load_image(file_path: str) -> np.ndarray:
    """
        Load an image from the specified file path and convert it to RGB format.
        Input: relative or absolute file path of the image
        Output: np.ndarray, each element type is np.float32
    """
    image_bgr = cv2.imread(file_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at {file_path}")
    image_np = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)

    image_np = image_np.astype(np.float32) / 255.0
    return image_np

def preprocess_image(image: np.ndarray, target_size = (256,256)) -> np.ndarray:
    """
    Resize and prepare the image for pose estimation models.

    Inputs:
        image (np.ndarray): Input RGB image array
        target_size (tuple): Desired (width, height)

    Outputs:
        np.ndarray: Preprocessed, resized image
    """
    resized = cv2.resize(image,target_size)
    return resized