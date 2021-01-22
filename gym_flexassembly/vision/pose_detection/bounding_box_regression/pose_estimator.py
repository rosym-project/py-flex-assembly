import joblib
import numpy as np

from gym_flexassembly.vision.pose_detection.bounding_box_regression.extract_features import detect_features

TRANLSATION_MODEL_ARG = 'translation_model'

class TranslationEstimator():

    def __init__(self, args):
        self.model = joblib.load(getattr(args, TRANLSATION_MODEL_ARG))

    def estimate(self, img):
        features = detect_features(img)
        features = np.array([features])
        return self.model.predict(features)[0]

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(f'--{TRANLSATION_MODEL_ARG}', type=str, required=True)
        return parser
