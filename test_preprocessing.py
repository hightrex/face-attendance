import numpy as np
import matplotlib.pyplot as plt

from smoke_test_data import (
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
)

from src.classical import (
    preprocess_image,
    preprocess_batch,
    extract_hog_features,
    flatten_images,
    fit_pca_features,
    transform_pca_features,
    ClassicalModel
)
# --------------------------------------------------
# TEST 1: Preprocess one image
# --------------------------------------------------

print("TEST 1: SINGLE IMAGE")
print("--------------------")

original_image = X_train[0]

print("Original image shape:")
print(original_image.shape)

processed_image = preprocess_image(original_image)

print("Processed image shape:")
print(processed_image.shape)

print("Processed image data type:")
print(processed_image.dtype)

print()


# --------------------------------------------------
# TEST 2: Preprocess all training images
# --------------------------------------------------

print("TEST 2: TRAINING BATCH")
print("----------------------")

processed_train = preprocess_batch(X_train)

print("Before preprocessing:")
print(X_train.shape)

print("After preprocessing:")
print(processed_train.shape)

print()


# --------------------------------------------------
# TEST 3: Validation images
# --------------------------------------------------

print("TEST 3: VALIDATION BATCH")
print("------------------------")

processed_val = preprocess_batch(X_val)

print("Before preprocessing:")
print(X_val.shape)

print("After preprocessing:")
print(processed_val.shape)

print()


# --------------------------------------------------
# TEST 4: Test images
# --------------------------------------------------

print("TEST 4: TEST BATCH")
print("------------------")

processed_test = preprocess_batch(X_test)

print("Before preprocessing:")
print(X_test.shape)

print("After preprocessing:")
print(processed_test.shape)

print()


# --------------------------------------------------
# TEST 5: Check pixel values
# --------------------------------------------------

print("TEST 5: PIXEL VALUES")
print("--------------------")

print("Minimum value:")
print(processed_train.min())

print("Maximum value:")
print(processed_train.max())

print()

# --------------------------------------------------
# TEST 6: HOG FEATURE EXTRACTION
# --------------------------------------------------

print()
print("TEST 6: HOG FEATURES")
print("--------------------")

hog_train = extract_hog_features(processed_train)
hog_val = extract_hog_features(processed_val)
hog_test = extract_hog_features(processed_test)

print("Training image shape:")
print(processed_train.shape)

print("HOG training feature shape:")
print(hog_train.shape)

print()

print("Validation HOG shape:")
print(hog_val.shape)

print()

print("Test HOG shape:")
print(hog_test.shape)

print()

assert hog_train.shape[0] == 30
assert hog_val.shape[0] == 9
assert hog_test.shape[0] == 9

assert hog_train.ndim == 2

print("All HOG tests passed successfully!")

# --------------------------------------------------
# TEST 7: PCA / EIGENFACE FEATURES
# --------------------------------------------------

print()
print("TEST 7: PCA FEATURES")
print("--------------------")

# Fit the scaler and PCA using TRAINING data only.
scaler, pca, pca_train = fit_pca_features(
    processed_train,
    n_components=20
)

# Validation and test data only use transform.
pca_val = transform_pca_features(
    processed_val,
    scaler,
    pca
)

pca_test = transform_pca_features(
    processed_test,
    scaler,
    pca
)

print("Original training shape:")
print(processed_train.shape)

print()

print("Flattened training shape:")
print(flatten_images(processed_train).shape)

print()

print("PCA training shape:")
print(pca_train.shape)

print()

print("PCA validation shape:")
print(pca_val.shape)

print()

print("PCA test shape:")
print(pca_test.shape)

print()

print("Variance explained by PCA:")
print(pca.explained_variance_ratio_.sum())

print()

# --------------------------------------------------
# TEST 8: SVM CLASSIFIER
# --------------------------------------------------

print()
print("TEST 8: SVM CLASSIFIER")
print("----------------------")


# --------------------------------------------------
# HOG + SVM
# --------------------------------------------------

print()
print("HOG + SVM")

hog_model = ClassicalModel(
    feature_type="hog",
    image_size=(64, 64),
    C=10.0
)

hog_model.fit(
    X_train,
    y_train
)

hog_probabilities = hog_model.predict_proba(
    X_val
)

print("Classes:")
print(hog_model.classes_)

print()

print("Probability table shape:")
print(hog_probabilities.shape)

print()

print("First validation probability row:")
print(hog_probabilities[0])


# Turn probabilities into predicted names.
hog_predictions = np.array(
    hog_model.classes_
)[
    np.argmax(
        hog_probabilities,
        axis=1
    )
]

hog_accuracy = np.mean(
    hog_predictions == y_val
)

print()

print("HOG validation predictions:")
print(hog_predictions)

print()

print("HOG validation accuracy:")
print(hog_accuracy)


# Check that the output has:
# 9 validation images and 3 possible people.
assert hog_probabilities.shape == (9, 3)

# Each probability row should add up to approximately 1.
assert np.allclose(
    hog_probabilities.sum(axis=1),
    1.0
)


# --------------------------------------------------
# PCA + SVM
# --------------------------------------------------

print()
print("PCA + SVM")

pca_model = ClassicalModel(
    feature_type="pca",
    image_size=(64, 64),
    C=10.0,
    pca_components=20
)

pca_model.fit(
    X_train,
    y_train
)

pca_probabilities = pca_model.predict_proba(
    X_val
)

print("Classes:")
print(pca_model.classes_)

print()

print("Probability table shape:")
print(pca_probabilities.shape)

print()

print("First validation probability row:")
print(pca_probabilities[0])


# Turn probabilities into predicted names.
pca_predictions = np.array(
    pca_model.classes_
)[
    np.argmax(
        pca_probabilities,
        axis=1
    )
]

pca_accuracy = np.mean(
    pca_predictions == y_val
)

print()

print("PCA validation predictions:")
print(pca_predictions)

print()

print("PCA validation accuracy:")
print(pca_accuracy)


assert pca_probabilities.shape == (9, 3)

assert np.allclose(
    pca_probabilities.sum(axis=1),
    1.0
)


print()
print("All SVM tests passed successfully!")


# Automatic PCA checks

assert pca_train.shape == (30, 20)
assert pca_val.shape == (9, 20)
assert pca_test.shape == (9, 20)

print("All PCA tests passed successfully!")


# --------------------------------------------------
# Automatic checks
# --------------------------------------------------

assert processed_train.shape == (30, 64, 64)
assert processed_val.shape == (9, 64, 64)
assert processed_test.shape == (9, 64, 64)

assert processed_train.dtype.name == "float32"

print("All preprocessing tests passed successfully!")

plt.imshow(processed_train[0], cmap="gray")
plt.title("Preprocessed Training Image")
plt.axis("off")
plt.show()