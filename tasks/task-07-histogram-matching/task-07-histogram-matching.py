# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    def match_channel(source, reference):
        # Flatten and get histogram and CDF
        src_hist, bins = np.histogram(source.flatten(), bins=256, range=[0, 256], density=True)
        ref_hist, _ = np.histogram(reference.flatten(), bins=256, range=[0, 256], density=True)
        
        src_cdf = np.cumsum(src_hist)
        ref_cdf = np.cumsum(ref_hist)

        # Normalizar CDFs
        src_cdf_norm = src_cdf / src_cdf[-1]
        ref_cdf_norm = ref_cdf / ref_cdf[-1]

        # Criar mapeamento de tons
        mapping = np.zeros(256, dtype=np.uint8)
        for src_val in range(256):
            diff = np.abs(ref_cdf_norm - src_cdf_norm[src_val])
            mapping[src_val] = np.argmin(diff)

        # Aplicar mapeamento
        matched = mapping[source]
        return matched

    # Separar canais R, G e B
    matched_img = np.zeros_like(source_img)
    for c in range(3):
        matched_img[:, :, c] = match_channel(source_img[:, :, c], reference_img[:, :, c])

    return matched_img
