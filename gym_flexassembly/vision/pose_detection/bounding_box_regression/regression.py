import argparse
import math
import sys
from enum import Enum

import numpy as np
from joblib import dump, load

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

import sklearn.metrics as metrics

class Model(Enum):
    SVR = 1
    DECISION_TREE_REGRESSOR = 2
    LASSO = 3
    ELASTIC_NET = 4

class Regressor():
    def __init__(self):
        self.reg = None
        self.model_type = None
        self.hyperparameters = None

    def load(self, model_path):
        self.reg = load(model_path)

    def save(self, model_path):
        dump(self.reg, model_path)

    def predict(self, x):
        if self.reg is None:
            print("error: unconfigured model")
            return

        """
        quaternions = self.reg.predict(x)
        lenghts = np.linalg.norm(quaternions, axis=1, keepdims=True)
        return quaternions / lenghts
        """
        return self.reg.predict(x)
        #"""

    def config(self, model_type, model_params):
        self.model_type = model_type
        self.hyperparameters = model_params

        if model_type == Model.SVR:
            if not model_params is None:
                self.reg = make_pipeline(StandardScaler(), MultiOutputRegressor(
                    SVR(epsilon=0.0005, max_iter=200000,
                        C=model_params["multioutputregressor__estimator__C"],
                        gamma=model_params["multioutputregressor__estimator__gamma"])))
            else:
                self.reg = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(epsilon=0.0005, max_iter=200000)))
        elif model_type == Model.DECISION_TREE_REGRESSOR:
            if not model_params is None:
                self.reg = make_pipeline(StandardScaler(),
                    DecisionTreeRegressor(max_depth=model_params["decisiontreeregressor__max_depth"],
                        min_samples_split=model_params["decisiontreeregressor__min_samples_split"]))
            else:
                self.reg = make_pipeline(StandardScaler(), DecisionTreeRegressor())
        elif model_type == Model.LASSO:
            if not model_params is None:
                self.reg = make_pipeline(StandardScaler(), MultiTaskLasso(alpha=model_params['multitasklasso__alpha']))
            else:
                self.reg = make_pipeline(StandardScaler(), MultiTaskLasso())
        elif model_type == Model.ELASTIC_NET:
            if not model_params is None:
                self.reg = make_pipeline(StandardScaler(), MultiTaskElasticNet(
                        alpha=model_params['multitaskelasticnet__alpha'], l1_ratio=model_params['multitaskelasticnet__l1_ratio']))
            else:
                self.reg = make_pipeline(StandardScaler(), MultiTaskElasticNet())

    def fit(self, x, y, n_jobs):
        if self.reg is None:
            print("error: unconfigured model")
            return

        cv_results = cross_validate(self.reg, x, y, n_jobs=n_jobs, return_estimator=True, scoring='neg_mean_squared_error')
        self.reg = cv_results['estimator'][np.argmax(cv_results['test_score'])]

    def train_hyperparameters(self, x, y, n_jobs):
        if self.model_type is None:
            print("error: unconfigured model")
            return

        if self.model_type == Model.SVR:
            parameters = {'multioutputregressor__estimator__C':[0.1, 1, 5, 10],
                        'multioutputregressor__estimator__gamma':[0.001, 0.01, 0.1, 1]}
        elif self.model_type == Model.DECISION_TREE_REGRESSOR:
            parameters = {'decisiontreeregressor__max_depth':[2, 5, 10, 15, 20, 25, None],
                        'decisiontreeregressor__min_samples_split':[2, 5, 10, 15, 20, 25]}
        elif self.model_type == Model.LASSO:
            parameters = {'multitasklasso__alpha':[1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1]}
        elif self.model_type == Model.ELASTIC_NET:
            parameters = {'multitaskelasticnet__alpha':[1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1], 'multitaskelasticnet__l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9]}

        grid = GridSearchCV(self.reg, parameters, n_jobs=n_jobs)
        grid.fit(x, y)
        self.hyperparameters = grid.best_params_

        print("Best hyperparameters:", self.hyperparameters)

        # reconfigure the model using the hyperparameters
        self.config(self.model_type, self.hyperparameters)


def main(args):
    parser = argparse.ArgumentParser('Trains and evaluates a support vector regression on a dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_train', type=str, default='./low_var_dataset/low_var/train',
                        help='the directory of the training dataset')
    parser.add_argument('--data_val', type=str, default='./low_var_dataset/low_var/val',
                        help='the directory of the validation dataset')
    parser.add_argument('--model_path', type=str, default=None,
                        help='the file path of where the trained model should be saved (file extension ".joblib")')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='number of parallel jobs (-1 means using all cpu cores)')
    args = parser.parse_args(args[1:])
    print(args)


    import csv
    import os
    with open(os.path.join(args.data_train + '/features.csv'), mode='r') as csv_file:
        feature_headers = next(csv.reader(csv_file))
        feature_headers = dict([(name, column) for column, name in enumerate(feature_headers)])

    with open(os.path.join(args.data_train + '/data.csv'), mode='r') as csv_file:
        data_headers = next(csv.reader(csv_file))
        data_headers = dict([(name, column) for column, name in enumerate(data_headers)])

    translation_cols = [data_headers[key] for key in data_headers if key in ['x', 'y', 'z']]
    orientation_cols = [data_headers[key] for key in data_headers if key in ['qx', 'qy', 'qz', 'qw']]
    cam_orientation_cols = [data_headers[key] for key in data_headers if key in ['camera_qx', 'camera_qy', 'camera_qz', 'camera_qw']]

    features_all = ['bb_center_x',
                    'bb_center_y',
                    'bb_height',
                    'bb_width',
                    'bb_angle[radians]',
                    'bb_area',
                    'laplacian_side',
                    'laplacian_sum',
                    'laplacian_abs_sum',
                    'laplacian_count']

    # load the datasets
    #"""
    # predict translation
    translation_features = features_all[:7]
    translation_features_cols = [feature_headers[key] for key in feature_headers if key in translation_features]

    print(f'Use features {translation_features}')

    data_train_x = np.loadtxt(args.data_train + "/features.csv", skiprows=1, usecols=translation_features_cols, delimiter=',')
    data_train_y = np.loadtxt(args.data_train + "/data.csv", skiprows=1, usecols=translation_cols, delimiter=',')
    data_val_x = np.loadtxt(args.data_val + "/features.csv", skiprows=1, usecols=translation_features_cols, delimiter=',')
    data_val_y = np.loadtxt(args.data_val + "/data.csv", skiprows=1, usecols=translation_cols, delimiter=',')
    """
    # predict rotation
    rotation_features = ['bb_height', 'bb_angle[radians]', 'laplacian_side', 'laplacian_abs_sum']
    rotation_features_cols = [feature_headers[key] for key in feature_headers if key in rotation_features]

    print(f'Use features {rotation_features}')

    data_train_x = np.c_[np.loadtxt(args.data_train + "/features.csv", skiprows=1, usecols=rotation_features_cols, delimiter=','),
                        np.loadtxt(args.data_train + "/data.csv", skiprows=1, usecols=cam_orientation_cols, delimiter=',')]
    data_train_y = np.loadtxt(args.data_train + "/data.csv", skiprows=1, usecols=orientation_cols, delimiter=',')
    data_val_x = np.c_[np.loadtxt(args.data_val + "/features.csv", skiprows=1, usecols=rotation_features_cols, delimiter=','),
                    np.loadtxt(args.data_val + "/data.csv", skiprows=1, usecols=cam_orientation_cols, delimiter=',')]
    data_val_y = np.loadtxt(args.data_val + "/data.csv", skiprows=1, usecols=orientation_cols, delimiter=',')
    """

    # setup and train the model
    reg = Regressor()
    reg.config(Model.SVR, None)
    reg.train_hyperparameters(data_train_x, data_train_y, args.n_jobs)
    reg.fit(data_train_x, data_train_y, args.n_jobs)

    # evaluate the model
    prediction = reg.predict(data_val_x)
    print("MSE:", metrics.mean_squared_error(data_val_y, prediction))
    print("MAE:", metrics.mean_absolute_error(data_val_y, prediction))

    if not args.model_path is None:
        reg.save(args.model_path)

if __name__ == '__main__':
    main(sys.argv)
