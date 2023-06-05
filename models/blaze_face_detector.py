# mediapipe face detector: https://developers.google.com/mediapipe/solutions/vision/face_detector

from .abstract.abstract_model import AbstractModel

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
from PIL import Image
import cv2


class FaceDetector(AbstractModel):

    def __init__(self,
                 model_weight: str,
                 min_detection_confidence: float = 0.5,
                 min_suppression_threshold: float = 0.3):
        self.model_weight = model_weight
        self.base_options = python.BaseOptions(model_asset_path=self.model_weight)
        self.options = vision.FaceDetectorOptions(
            base_options=self.base_options,
            min_detection_confidence=min_detection_confidence,
            min_suppression_threshold=min_suppression_threshold)
        self.model = vision.FaceDetector.create_from_options(self.options)

        # this might make it easier to get the image for the inference result
        self._image_key = 'img'

    def __repr__(self):
        return 'MediaPipe BlazeFace: https://developers.google.com/mediapipe/solutions/vision/face_detector'

    def preprocess(self, rgb_image: np.ndarray) -> mp.Image:
        """Convert RGB image to mediapipe Image

        Args:
            rgb_image (np.ndarray): an image in numpy array (H, W, 3) RGB

        Returns:
            mp.Image: _description_
        """

        # handle PIL Image
        image_arr = np.asarray(rgb_image)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
        return mp_img

    def __crop_image(self, image: np.array, x, y, w, h) -> np.ndarray:
        return image[y:y + h, x:x + w, :]

    def postprocess(self, detections, mp_image) -> dict:
        """Take mediapipe prediction and image
        Convert it to human-readable format

        Return Cropped images in RGB format

        Args:
            detections (_type_): _description_
            mp_image (_type_): _description_

        Returns:
            dict: _description_
        """

        faces = []
        image_arr = mp_image.numpy_view()
        detections = detections.detections

        for i, detection in enumerate(detections):
            bbox = detection.bounding_box
            x, y = bbox.origin_x, bbox.origin_y
            w, h = bbox.width, bbox.height
            face_area = self.__crop_image(image_arr, x, y, w, h)

            faces.append({
                'index': i,
                self._image_key: face_area,
                'coord': {
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                }
            })

        return faces

    def __call__(self, image: Image.Image | np.ndarray) -> dict:
        """Run inference

        Args:
            image (Image.Image | np.ndarray): an image in (h, w, 3) RGB

        Returns:
            dict: _description_
        """
        mp_img = self.preprocess(image)
        detections = self.model.detect(mp_img)
        faces = self.postprocess(detections, mp_img)

        return faces

    # alias
    detect = predict = __call__