import argparse
import math
import sys

import numpy as np
from joblib import dump, load

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import sklearn.metrics as metrics

def main(args):
    parser = argparse.ArgumentParser('Trains and evaluates a support vector regression on a dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_train', type=str, default='./low_var_dataset/low_var/train',
                        help='the directory of the training dataset')
    parser.add_argument('--data_val', type=str, default='./low_var_dataset/low_var/val',
                        help='the directory of the validation dataset')
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args(args[1:])
    print(args)

    # load the datasets
    data_train_x = np.loadtxt(args.data_train + "/features.csv", skiprows=1, usecols=[2, 3, 4, 5, 6, 7], delimiter=',')
    data_train_y = np.loadtxt(args.data_train + "/data.csv", skiprows=1, usecols=[2, 3, 4], delimiter=',')
    #data_train_y = np.loadtxt(args.data_train + "/data.csv", skiprows=1, usecols=[5, 6, 7], delimiter=',')
    data_val_x = np.loadtxt(args.data_val + "/features.csv", skiprows=1, usecols=[2, 3, 4, 5, 6, 7], delimiter=',')
    data_val_y = np.loadtxt(args.data_val + "/data.csv", skiprows=1, usecols=[2, 3, 4], delimiter=',')
    #data_val_y = np.loadtxt(args.data_val + "/data.csv", skiprows=1, usecols=[5, 6, 7], delimiter=',')

    # train the model
    reg = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR()))
    #reg = make_pipeline(StandardScaler(), DecisionTreeRegressor())
    reg.fit(data_train_x, data_train_y)
    #reg = load(args.model_path + "/model.joblib")

    # evaluate the model
    prediction = reg.predict(data_val_x)
    print("MSE:", metrics.mean_squared_error(data_val_y, prediction))
    print("MAE:", metrics.mean_absolute_error(data_val_y, prediction))

    if not args.model_path is None:
        dump(reg, args.model_path + "/model.joblib")

if __name__ == '__main__':
    main(sys.argv)
