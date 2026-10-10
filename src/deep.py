"""
src/deep.py
Deep learning model for the face attendance project.
Viraj - CSI6209 Assessment 2

This file turns face photos into embeddings using a pretrained FaceNet
network, then trains a simple Logistic Regression classifier on top.
"""

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from sklearn.linear_model import LogisticRegression
from PIL import Image
from torchvision import transforms


def preprocess_for_facenet(images, image_size=160):
    """
    Prepare a list of images so they can go into the FaceNet model.

    FaceNet wants:
    - RGB images
    - size 160x160
    - pixel values scaled in a specific way (fixed_image_standardization)
    """
    processed = []

    for img in images:
        img = np.asarray(img)

        # Make sure it is RGB
        if img.ndim == 2:                       # grayscale
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 4:  # RGBA
            img = img[:, :, :3]
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.concatenate([img] * 3, axis=-1)

        # Convert to 0-255 range if the image is in 0-1
        if img.dtype in (np.float32, np.float64):
            if img.max() <= 1.01:
                img = (img * 255.0).astype(np.float32)
            else:
                img = img.astype(np.float32)
        else:
            img = img.astype(np.float32)

        # Resize with PIL
        pil_img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)

        # Convert to tensor and apply the official FaceNet scaling
        tensor = transforms.ToTensor()(pil_img)
        tensor = fixed_image_standardization(tensor * 255.0)

        processed.append(tensor)

    return torch.stack(processed)


class DeepModel:
    """
    Main deep learning model for the project.

    Steps:
    1. Load a pretrained FaceNet network (InceptionResnetV1 trained on VGGFace2)
    2. Turn every face photo into a 512-number embedding
    3. Train a Logistic Regression classifier on those embeddings
    """

    def __init__(self, C=1.0, batch_size=32, device=None):
        self.C = C
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.resnet = None
        self.classifier = None
        self.classes_ = None

    def _get_resnet(self):
        """Load the pretrained network the first time we need it."""
        if self.resnet is None:
            print("Loading pretrained FaceNet (VGGFace2)...")
            self.resnet = InceptionResnetV1(pretrained="vggface2")
            self.resnet.eval()
            self.resnet = self.resnet.to(self.device)
        return self.resnet

    def _get_embeddings(self, X):
        """Run images through FaceNet and return the embeddings."""
        resnet = self._get_resnet()
        all_embeddings = []

        for i in range(0, len(X), self.batch_size):
            batch = X[i : i + self.batch_size]
            batch_tensor = preprocess_for_facenet(batch).to(self.device)

            with torch.no_grad():
                emb = resnet(batch_tensor)

            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)

    def fit(self, X, y):
        """Train the model."""
        print("Computing FaceNet embeddings for training images...")
        embeddings = self._get_embeddings(X)

        print("Training Logistic Regression on the embeddings...")
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
        self.classifier.fit(embeddings, y)
        self.classes_ = list(self.classifier.classes_)
        return self

    def predict_proba(self, X):
        """Return probability for each person for every image."""
        if self.classifier is None:
            raise RuntimeError("You must call fit() before predict_proba()")
        embeddings = self._get_embeddings(X)
        return self.classifier.predict_proba(embeddings)

    def predict(self, X):
        """Return the most likely person name for each image."""
        probs = self.predict_proba(X)
        best = np.argmax(probs, axis=1)
        return np.array(self.classes_)[best]