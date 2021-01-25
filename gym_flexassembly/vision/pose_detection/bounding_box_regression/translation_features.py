import argparse
import math
import sys

import numpy as np

from regression import Regressor


def main(args):
    parser = argparse.ArgumentParser('Estimates the translation of clamps and adds it as an additional feature',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str, default='./low_var_dataset/low_var/train',
                        help='the directory of the dataset')
    parser.add_argument('-r', '--regressor', type=str, default='./translation_regressor.joblib',
                        help='the model file of the regressor')
    args = parser.parse_args(args[1:])
    print(args)

    data = np.loadtxt(args.data_dir + "/features.csv", dtype=np.str, delimiter=",")
    header = np.append(data[0], ["translation_x", "translation_y", "translation_z"])
    data = data[1:]

    reg = Regressor()
    reg.load(args.regressor)
    data = np.c_[data, reg.predict(data[:, 2:8])]
    np.savetxt(args.data_dir + "/features.csv", data, fmt='%s', delimiter=',', header=','.join(header))

if __name__ == '__main__':
    main(sys.argv)
