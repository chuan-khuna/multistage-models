import numpy as np
from PIL import Image
import json


class BundleModel:

    def __init__(self, face_detector):
        """A multi-stage face detection models

        The first stage is detecting faces in an input
        The following stages will do something with those detected faces

        Args:
            face_detector (_type_): face_detector should be callable via `face_detector(image)`
        """

        self.face_detector = face_detector
        self.models = {}

    def add(self, name, model):
        """Add a model the the following stage after detecting faces

        Args:
            name (_type_): stage name/model name
            model (_type_): callable model object
        """
        self.models[name] = model
        return self

    def __repr__(self) -> str:
        return f"Models: {self.models}"

    def __call__(self, image: Image.Image | np.ndarray) -> dict:
        """Run multi-stage model inference
        start from the face detection model

        Args:
            image (Image.Image | np.ndarray): an image in (h, w, 3) RGB

        Returns:
            dict: inference result
        """

        faces = self.face_detector(image)

        for model_name, model in self.models.items():
            for i, face in enumerate(faces):
                # the face detector return RGB numpy
                face_img: np.ndarray = face[self.face_detector._image_key]
                pred = model(face_img)
                faces[i][model_name] = pred

        return faces