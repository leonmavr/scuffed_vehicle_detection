#!/usr/env/bin python3

from hog import hog
from heatmap import Heatmap
from cv2 import cv2
import pickle
import os
import sys
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)


def load_classifier(fpath = os.path.join(this_script_folder, 'models', 'car_classifier_64x64.pkl')):
    with open(fpath, 'rb') as f:
        clf = pickle.load(f)
    return clf


if __name__ == '__main__':
    ### modify this section to change the demo properties
    # iterate every `coarseness` pixels before attempting to use the classifier
    coarseness = 8
    # strip of image starting from the bottom where we do not attempt to detect anything
    horizon = 0.4
    # pre-trained binary classifier for vehicle detection
    clf = load_classifier()
    # video data
    vid = os.path.join(this_script_folder, 'test_data', 'videos', 'udacity.mp4')
    ### do not modify anything below

    cap = cv2.VideoCapture(vid)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        valid, frame = cap.read()
        # the HOG feature detector expects a 2D matrix
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('camera', frame)
        c = cv2.waitKey(10)
        if c == ord('q') or not valid:
            break

    cap.release()
    cv2.destroyAllWindows()