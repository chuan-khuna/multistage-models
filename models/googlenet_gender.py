# https://github.com/onnx/models/blob/main/vision/body_analysis/age_gender/levi_googlenet.py
# https://github.com/onnx/models/tree/main/vision/body_analysis/age_gender

from .abstract.abstract_model import AbstractModel
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort


class GenderModel(AbstractModel):

    def __init__(self, model_weight):
        self.model_weight = model_weight
        self.model = ort.InferenceSession(self.model_weight)
        self.input_name = self.model.get_inputs()[0].name

        self.gender_list = ['Male', 'Female']

    def __repr__(self):
        return 'ONNX googlenet gender: https://github.com/onnx/models/tree/main/vision/body_analysis/age_gender'

    def preprocess(self, rgb_image: Image.Image | np.ndarray) -> np.ndarray:
        image = np.asarray(rgb_image)
        image = cv2.resize(image, (224, 224))
        image_mean = np.array([104, 117, 123])
        image = image - image_mean
        # (3-colour, h, w)
        image = np.transpose(image, [2, 0, 1])
        # (1-batch size, 3-colour, h, w)
        image = np.expand_dims(image, axis=0)
        return image.astype(np.float32)

    def postprocess(self, pred: np.ndarray) -> str:
        return self.gender_list[pred[0].argmax()]

    def __call__(self, data: Image.Image | np.ndarray) -> str:
        """Run inference

        Args:
            data (Image.Image | np.ndarray): an image in (h, w, 3) RGB

        Returns:
            str: _description_
        """

        image = self.preprocess(data)
        predictions = self.model.run(None, {self.input_name: image})
        gender = self.postprocess(predictions)
        return gender

    predict = __call__
