# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    def mse(img1, img2):
        return np.mean((img1 - img2) ** 2)

    def psnr(img1, img2):
        mse_val = mse(img1, img2)
        if mse_val == 0:
            return float('inf')
        return 10 * np.log10(1.0 / mse_val)

    def ssim(img1, img2):
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
        return numerator / denominator

    def npcc(img1, img2):
        mean1 = np.mean(img1)
        mean2 = np.mean(img2)
        numerator = np.sum((img1 - mean1) * (img2 - mean2))
        denominator = np.sqrt(np.sum((img1 - mean1) ** 2) * np.sum((img2 - mean2) ** 2))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    return {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }
