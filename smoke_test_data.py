import numpy as np


def make_image(person, size=80, noise=0.05, rng=None):
    """
    Create a simple fake RGB image for testing.

    person:
        0 = vertical shape on the left
        1 = vertical shape on the right
        2 = horizontal shape in the middle
    """

    if rng is None:
        rng = np.random.default_rng()

    # Start with a black grayscale image
    image = np.zeros((size, size), dtype=np.float32)

    # Give each fake person a different pattern
    if person == 0:
        image[15:65, 15:25] = 1.0

    elif person == 1:
        image[15:65, 55:65] = 1.0

    elif person == 2:
        image[35:45, 15:65] = 1.0

    # Add small random noise so every image is slightly different
    image += rng.normal(0, noise, image.shape)

    # Keep pixel values between 0 and 1
    image = np.clip(image, 0, 1)

    # Convert grayscale image into RGB
    image_rgb = np.stack([image, image, image], axis=-1)

    return image_rgb


def make_dataset(images_per_person, seed):
    """
    Create a dataset containing three fake people.
    """

    rng = np.random.default_rng(seed)

    images = []
    labels = []

    names = ["Person_A", "Person_B", "Person_C"]

    for person_number, person_name in enumerate(names):

        for _ in range(images_per_person):

            image = make_image(
                person=person_number,
                size=80,
                noise=0.05,
                rng=rng
            )

            images.append(image)
            labels.append(person_name)

    X = np.array(images)
    y = np.array(labels)

    return X, y


# -----------------------------
# Create temporary datasets
# -----------------------------

X_train, y_train = make_dataset(
    images_per_person=10,
    seed=1
)

X_val, y_val = make_dataset(
    images_per_person=3,
    seed=2
)

X_test, y_test = make_dataset(
    images_per_person=3,
    seed=3
)


# -----------------------------
# Display information
# -----------------------------

print("TRAINING DATA")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print()

print("VALIDATION DATA")
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

print()

print("TEST DATA")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print()

print("Training labels:")
print(y_train)