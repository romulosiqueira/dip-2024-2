import numpy as np
import cv2 as cv
import requests

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    response = requests.get(url)
    response.raise_for_status()  
    
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    
    image = cv.imdecode(image_array, **kwargs)
    
    return image

load_image_from_url(); 