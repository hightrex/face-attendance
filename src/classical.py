import numpy as np

from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# PREPROCESS ONE IMAGE
# --------------------------------------------------

def preprocess_image(image, image_size=(64, 64)):
    """
    Convert one image to grayscale and resize it.
    """

    # Make sure the input is a NumPy array.
    image = np.asarray(image)

    # If the image is RGB or RGBA.
    if image.ndim == 3:

        # Remove alpha channel if present.
        if image.shape[-1] == 4:
            image = image[:, :, :3]

        # Convert colour image to grayscale.
        grayscale_image = rgb2gray(image)

    # If already grayscale.
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

    # Keep data type consistent.
    resized_image = resized_image.astype(np.float32)

    return resized_image


# --------------------------------------------------
# PREPROCESS A BATCH OF IMAGES
# --------------------------------------------------

def preprocess_batch(images, image_size=(64, 64)):
    """
    Preprocess several images.

    Each image is converted to grayscale and resized.
    """

    processed_images = []

    for image in images:

        processed_image = preprocess_image(
            image,
            image_size=image_size
        )

        processed_images.append(processed_image)

    return np.array(
        processed_images,
        dtype=np.float32
    )


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

    return np.array(
        feature_vectors,
        dtype=np.float32
    )


# --------------------------------------------------
# FLATTEN IMAGES FOR PCA
# --------------------------------------------------

def flatten_images(images):
    """
    Flatten grayscale images into rows.

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
    Fit StandardScaler and PCA using training images only.

    Returns:
        scaler
        pca
        PCA training features
    """

    # Flatten images first.
    flattened_images = flatten_images(images)

    # Fit scaler using training data only.
    scaler = StandardScaler()

    scaled_images = scaler.fit_transform(
        flattened_images
    )

    # Fit PCA using training data only.
    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    features = pca.fit_transform(
        scaled_images
    )

    return scaler, pca, features


# --------------------------------------------------
# APPLY TRAINED PCA TO NEW IMAGES
# --------------------------------------------------

def transform_pca_features(images, scaler, pca):
    """
    Apply an already fitted scaler and PCA.

    This is used for validation or test images.
    """

    flattened_images = flatten_images(images)

    # Transform only.
    # Do not fit again.
    scaled_images = scaler.transform(
        flattened_images
    )

    features = pca.transform(
        scaled_images
    )

    return features.astype(np.float32)


# --------------------------------------------------
# CLASSICAL MACHINE LEARNING MODEL
# --------------------------------------------------

class ClassicalModel:
    """
    Classical machine learning model.

    It can use either:

    HOG + SVM

    or

    PCA/eigenfaces + SVM
    """

    def __init__(
        self,
        feature_type="hog",
        image_size=(64, 64),
        C=10.0,
        pca_components=20,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    ):

        # Only allow HOG or PCA.
        if feature_type not in ("hog", "pca"):
            raise ValueError(
                "feature_type must be either 'hog' or 'pca'."
            )

        self.feature_type = feature_type
        self.image_size = image_size
        self.C = C

        self.pca_components = pca_components

        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

        # These will be learned during training.
        self.scaler = None
        self.pca = None
        self.classifier = None

        # Stores the class/person names.
        self.classes_ = None


    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    def fit(self, X, y):
        """
        Train the model.

        X = training images
        y = matching person labels
        """

        # Step 1:
        # Preprocess the training images.
        processed_images = preprocess_batch(
            X,
            image_size=self.image_size
        )

        # --------------------------------------------------
        # HOG ROUTE
        # --------------------------------------------------

        if self.feature_type == "hog":

            features = extract_hog_features(
                processed_images,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block
            )

            # Scale HOG features.
            self.scaler = StandardScaler()

            features = self.scaler.fit_transform(
                features
            )

            # PCA is not used for HOG.
            self.pca = None


        # --------------------------------------------------
        # PCA ROUTE
        # --------------------------------------------------

        else:

            # PCA cannot have more components than
            # the number of training samples.
            max_components = min(
                processed_images.shape[0],
                processed_images.shape[1]
                * processed_images.shape[2]
            )

            if self.pca_components > max_components:

                raise ValueError(
                    f"pca_components={self.pca_components} "
                    f"is too large. Maximum allowed with "
                    f"this training set is {max_components}."
                )

            self.scaler, self.pca, features = fit_pca_features(
                processed_images,
                n_components=self.pca_components
            )


        # --------------------------------------------------
        # TRAIN SVM
        # --------------------------------------------------

        self.classifier = SVC(
            C=self.C,
            probability=True,
            random_state=42
        )

        self.classifier.fit(
            features,
            y
        )

        # Store class names in the same order used
        # by predict_proba().
        self.classes_ = self.classifier.classes_.tolist()

        return self


    # --------------------------------------------------
    # PREDICT PROBABILITIES
    # --------------------------------------------------

    def predict_proba(self, X):
        """
        Return probabilities for each person.

        Each row represents one image.
        Each column represents one person/class.
        """

        # Prevent prediction before training.
        if self.classifier is None:

            raise RuntimeError(
                "The model must be fitted before "
                "predict_proba() is used."
            )

        # Preprocess images using the same settings.
        processed_images = preprocess_batch(
            X,
            image_size=self.image_size
        )


        # --------------------------------------------------
        # HOG PREDICTION ROUTE
        # --------------------------------------------------

        if self.feature_type == "hog":

            features = extract_hog_features(
                processed_images,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block
            )

            # Important:
            # transform only, do not fit again.
            features = self.scaler.transform(
                features
            )


        # --------------------------------------------------
        # PCA PREDICTION ROUTE
        # --------------------------------------------------

        else:

            features = transform_pca_features(
                processed_images,
                self.scaler,
                self.pca
            )


        # Return probabilities from the SVM.
        probabilities = self.classifier.predict_proba(
            features
        )

        return probabilities