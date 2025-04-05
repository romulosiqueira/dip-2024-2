# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    h, w = img.shape

    # 1. Translate: shift right and down by 10 pixels
    translated = np.zeros_like(img)
    shift_y, shift_x = 10, 10
    translated[shift_y:, shift_x:] = img[:h-shift_y, :w-shift_x]

    # 2. Rotate 90 degrees clockwise
    rotated = np.rot90(img, k=-1)

    # 3. Stretch horizontally (scale width by 1.5)
    new_w = int(w * 1.5)
    stretched = np.zeros((h, new_w))
    for i in range(new_w):
        orig_x = int(i / 1.5)
        if orig_x < w:
            stretched[:, i] = img[:, orig_x]

    # 4. Mirror horizontally (flip along vertical axis)
    mirrored = img[:, ::-1]

    # 5. Barrel distortion (radial distortion)
    distorted = np.zeros_like(img)
    center_y, center_x = h / 2, w / 2
    for y in range(h):
        for x in range(w):
            norm_y = (y - center_y) / h
            norm_x = (x - center_x) / w
            r = np.sqrt(norm_x ** 2 + norm_y ** 2)
            factor = 1 + 0.3 * (r ** 2)  # distortion factor
            src_y = int(center_y + norm_y / factor * h)
            src_x = int(center_x + norm_x / factor * w)
            if 0 <= src_y < h and 0 <= src_x < w:
                distorted[y, x] = img[src_y, src_x]

    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }
