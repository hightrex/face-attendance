from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from .preprocessing import normalize_images


def load_data():
    faces = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
    x = normalize_images(faces.data)
    y = faces.target
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.25,
        random_state=42, stratify=y_train_val
    )
    return x_train, x_val, x_test, y_train, y_val, y_test, faces.target_names, faces.images.shape[1:]
