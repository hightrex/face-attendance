import numpy as np

from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# PREPROCESS ONE IMAGE
# --------------------------------------------------

def preprocess_image(image, image_size=(64, 64)):
    """
    Convert one image to grayscale and resize it.

    Parameters
    ----------
    image : numpy.ndarray
        One input image.

    image_size : tuple
        Desired output size as (height, width).

    Returns
    -------
    numpy.ndarray
        A grayscale resized image.
    """

    # Make sure the input is a NumPy array.
    image = np.asarray(image)

    # If the image is RGB, convert it to grayscale.
    if image.ndim == 3:

        # If an image has four channels (RGBA), remove alpha.
        if image.shape[-1] == 4:
            image = image[:, :, :3]

        grayscale_image = rgb2gray(image)

    # If already grayscale, keep it as grayscale.
    elif image.ndim == 2:
        grayscale_image = image

    else:
        raise ValueError(
            "Image must be either a 2D grayscale image "
            "or a 3D RGB image."
        )

    # Resize the grayscale image.
    resized_image = resize(
        grayscale_image,
        image_size,
        anti_aliasing=True
    )

    # Keep the data type consistent.
    resized_image = resized_image.astype(np.float32)

    return resized_image


# --------------------------------------------------
# PREPROCESS A BATCH OF IMAGES
# --------------------------------------------------

def preprocess_batch(images, image_size=(64, 64)):
    """
    Preprocess a collection of images.

    Each image is converted to grayscale and resized.
    """

    processed_images = []

    for image in images:

        processed_image = preprocess_image(
            image,
            image_size=image_size
        )

        processed_images.append(processed_image)

    return np.array(processed_images, dtype=np.float32)


# --------------------------------------------------
# HOG FEATURE EXTRACTION
# --------------------------------------------------

def extract_hog_features(
    images,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2)
):
    """
    Extract HOG features from preprocessed grayscale images.
    """

    feature_vectors = []

    for image in images:

        features = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True
        )

        feature_vectors.append(features)

    return np.array(feature_vectors, dtype=np.float32)


# --------------------------------------------------
# FLATTEN IMAGES FOR PCA
# --------------------------------------------------

def flatten_images(images):
    """
    Flatten grayscale images into rows of pixel values.

    Example:
    (30, 64, 64) becomes (30, 4096).
    """

    images = np.asarray(images)

    flattened_images = images.reshape(
        images.shape[0],
        -1
    )

    return flattened_images


# --------------------------------------------------
# FIT PCA USING TRAINING DATA
# --------------------------------------------------

def fit_pca_features(images, n_components=20):
    """
    Fit a scaler and PCA using training images only.

    Returns
    -------
    scaler
        StandardScaler fitted on training data.

    pca
        PCA fitted on training data.

    features
        PCA representation of training images.
    """

    # Convert every image into one row.
    flattened_images = flatten_images(images)

    # Fit the scaler on TRAINING data only.
    scaler = StandardScaler()

    scaled_images = scaler.fit_transform(
        flattened_images
    )

    # Fit PCA on TRAINING data only.
    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    features = pca.fit_transform(
        scaled_images
    )

    return scaler, pca, features


# --------------------------------------------------
# APPLY PCA TO VALIDATION OR TEST DATA
# --------------------------------------------------

def transform_pca_features(images, scaler, pca):
    """
    Apply an already-fitted scaler and PCA to new images.

    The scaler and PCA are NOT fitted again here.
    """

    flattened_images = flatten_images(images)

    # Only transform validation/test data.
    scaled_images = scaler.transform(
        flattened_images
    )

    features = pca.transform(
        scaled_images
    )

    return features.astype(np.float32)