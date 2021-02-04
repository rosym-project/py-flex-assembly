import joblib
import numpy as np

from gym_flexassembly.vision.pose_detection.bounding_box_regression.extract_features import detect_features

TRANLSATION_MODEL_ARG = 'translation_model'
ROTATION_MODEL_ARG = 'rotation_model'

class TranslationEstimator():

    def __init__(self, args):
        self.model = joblib.load(getattr(args, TRANLSATION_MODEL_ARG))

    def estimate(self, img):
        _, features = detect_features(img)
        features = np.array([features])
        return self.model.predict(features[:, [0,1,2,3,4,6]])[0]

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(f'--{TRANLSATION_MODEL_ARG}', type=str, required=True)
        return parser

class RotationEstimator():

    def __init__(self, args):
        self.model = joblib.load(getattr(args, ROTATION_MODEL_ARG))

    def estimate(self, img, translation, camera_orientation):
        # assemble feature vector
        _, features = detect_features(img)
        features = np.array(features)
        features = np.hstack((features, translation, camera_orientation))

        # not all features are used by the current model
        used_features = [2, 4, 6, 8, 13, 14, 15, 16]

        # predict and normalize quaternion
        quaternion = self.model.predict([features[used_features]])[0]
        quaternion /= np.linalg.norm(quaternion)
        return quaternion

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(f'--{ROTATION_MODEL_ARG}', type=str, required=True)
        return parser
