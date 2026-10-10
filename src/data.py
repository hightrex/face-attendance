"""
src/data.py
LFW data loader and stratified train/val/test splits
Tshering - CSI6209 Assessment 2
"""

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(min_faces_per_person=20, resize=0.4, random_state=42):
    """
    Loads the LFW dataset and splits it into train, validation and test sets.
    
    Returns the image arrays and the person names (as strings).
    """

    # download the dataset (only people with at least min_faces_per_person photos)
    # color=False so we get grayscale images
    faces = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=resize,
        color=False
    )

    # faces.images has the actual pictures (height x width)
    # faces.data is flattened, we don't want that
    X = faces.images
    y_int = faces.target
    target_names = faces.target_names

    # convert the number labels into actual names
    y = target_names[y_int]

    print(f"Loaded LFW: {len(target_names)} people, {len(X)} images")
    print(f"Image shape: {X.shape[1:]}")
    print(f"min_faces_per_person = {min_faces_per_person}")

    # first split off the test set (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

    # now split the remaining 80% into train (60%) and validation (20%)
    # 0.25 * 0.80 = 0.20, so we end up with 60/20/20
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_val
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        target_names,
        X.shape[1:]
    )


if __name__ == "__main__":
    # just to quickly check that everything works
    X_train, X_val, X_test, y_train, y_val, y_test, names, shape = load_data()
    print("Sample names:", names[:5])
    print("First training label:", y_train[0])
    print("First image shape:", X_train[0].shape)