import numpy as np

def normalize_images(images):
    """Convert grayscale pixel intensities to floating point values in [0, 1]."""
    images = np.asarray(images, dtype=np.float32)
    return images / 255.0 if images.max() > 1.0 else images
