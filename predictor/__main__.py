"""A command line control for the predictor."""
import argparse
import tensorflow as tf
from predictor import Predictor
from collector import *


def main(FLAGS):
    with tf.Session() as sess:
        for i in range(2003, 2015):
            print("Predicint for year {}".format(i))
            p = Predictor()
            p.train(i)


if __name__ == "__main__":
    """The main point of entry into the predictor"""
    parser = argparse.ArgumentParser()
    parser.add_argument("predict",
                        type=str,
                        help="Predict values for target years in a folder")
    args = parser.parse_args()
    main(args)
