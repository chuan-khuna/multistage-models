from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

from models.blaze_face_detector import FaceDetector
from models.mediapipe_image_embedding import EmbeddingModel
from models.multi_stage_face_model import BundleModel
from models.googlenet_age import AgeModel
from models.googlenet_gender import GenderModel

from utils.cv_utils import cv_rectangle

import cProfile
import pstats


def main():
    IMAGE_PATH = 'data/cover-76.jpg'

    img = Image.open(IMAGE_PATH)  # RGB Image.Image
    img_cv = cv2.imread(IMAGE_PATH)  # BGR numpy array

    detector = FaceDetector('weights/blaze_face_short_range.tflite',
                            min_detection_confidence=0.6,
                            min_suppression_threshold=0.4)
    embedder = EmbeddingModel('weights/mobilenet_v3_small.tflite')
    age_model = AgeModel('weights/age_googlenet.onnx')
    gender_model = GenderModel('weights/gender_googlenet.onnx')

    bundle = BundleModel(detector)
    bundle.add(name='embedding', model=embedder)
    bundle.add(name='age', model=age_model)
    bundle.add(name='gender', model=gender_model)

    faces = bundle(img)

    print(f"Detected faces: {len(faces)}")
    print('---')

    for face in faces:
        vect = face['embedding']

        for k, v in face.items():
            if k != bundle.face_detector._image_key:
                print(k, v)

        # plt.imshow(face[bundle.face_detector._image_key])
        # plt.axis('off')
        # plt.show()


if __name__ == '__main__':
    with cProfile.Profile() as profiler:
        main()
        profiler.dump_stats('stats.prof')
        with open('stats.txt', 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumtime')
            stats.print_stats()