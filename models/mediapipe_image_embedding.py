# https://developers.google.com/mediapipe/solutions/vision/image_embedder

from .abstract.abstract_model import AbstractModel

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
from PIL import Image
import cv2


class EmbeddingModel(AbstractModel):

    def __init__(self, model_weight: str):
        self.model_weight = model_weight
        self.base_options = python.BaseOptions(model_asset_path=self.model_weight)

        # set `l2_normalize`` and `quantize` to return a vector in range int (0..255)
        # this option will return float (-1..1)
        self.options = vision.ImageEmbedderOptions(base_options=self.base_options)
        self.model = vision.ImageEmbedder.create_from_options(self.options)

    def __repr__(self):
        return 'MediaPipe Embedding: https://developers.google.com/mediapipe/solutions/vision/image_embedder'

    def preprocess(self, rgb_image: np.ndarray) -> mp.Image:
        """Convert RGB image to mediapipe Image

        Args:
            rgb_image (np.ndarray): an image in numpy array (H, W, 3) RGB

        Returns:
            mp.Image: _description_
        """

        # handle PIL Image
        image_arr = np.asarray(rgb_image)
        image_arr = cv2.resize(image_arr, (224, 224))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
        return mp_img

    def postprocess(self, prediction) -> dict:
        return prediction.embeddings[0].embedding

    def __call__(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Embed an image to a vector

        Args:
            image (Image.Image | np.ndarray): an image in (h, w, 3) RGB

        Returns:
            np.ndarray: an embedding vector that represents the input image with shape `(n_dims, )`
        """
        mp_img = self.preprocess(image)
        pred = self.model.embed(mp_img)
        embedding_vector = self.postprocess(pred)

        return embedding_vector

    # alias
    embed = __call__