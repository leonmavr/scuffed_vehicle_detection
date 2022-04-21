#!/usr/env/bin python3

from hog import hog
from heatmap import Heatmap
from cv2 import cv2
import pickle
import os
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)


def load_classifier(fpath = os.path.join(this_script_folder, 'models',
                    'car_classifier_64x64.pkl')):
    with open(fpath, 'rb') as f:
        clf = pickle.load(f)
    return clf


if __name__ == '__main__':
    ### modify this section to change the demo properties
    # iterate every `coarseness` pixels before attempting to use the classifier
    coarseness = 8
    # strip of image starting from the bottom where we do attempt to perform detections
    horizon = 0.45
    assert 0 <= horizon <= 1, "The horizon lies from 0 to 1 relative to the bottom of the frame"
    # pre-trained binary classifier for vehicle detection
    clf = load_classifier()
    # video data
    vid = os.path.join(this_script_folder, 'test_data', 'videos', 'udacity.mp4')
    # the size of the image patches the HOG create features of
    patch_size_hog_x = 64
    patch_size_hog_y = 64
    # any region with hits (counts) less than this threshold will not be detected
    heatmap_threshold = 3
    bounding_box_color = (50, 255, 0) # as BGR

    ### do not modify anything below
    cap = cv2.VideoCapture(vid)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    heatmap = None

    while True:
        valid, frame = cap.read()
        if heatmap is None:
            heatmap = Heatmap((frame.shape[0], frame.shape[1]), heatmap_threshold)
        # the HOG feature detector expects a 2D matrix
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # crop it below the horizon
        rows, cols = frame.shape
        #frame = frame[int((1-horizon)*rows):, :]
        horizon = int((1-horizon)*rows)
        # bounding boxes to be refined if they contain a positive detection
        bboxes = []
        for x in range(0, cols-patch_size_hog_x, coarseness):
            for y in range(0, rows-patch_size_hog_y, coarseness):
                if y > horizon:
                    patch_to_detect = frame[y:y+patch_size_hog_y, x:x+patch_size_hog_x]
                    #print(patch_to_detect)
                    # TODO: 99 ->1
                    #if clf.predict([hog(patch_to_detect)])[0] == 1:
                    if False:
                        bboxes.append((x, x+patch_size_hog_x, y, y+patch_size_hog_y))
                        print(bboxes)
                    #clf.predict([hog(f_abs_path)])[0]
        bboxes = [(100, 200, 200, 300),
                  (125, 225, 200, 300),
                  (125, 225, 200, 300),
                  (125, 225, 200, 300)]
        bboxes = heatmap.refine(bboxes)
        for b in bboxes:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), bounding_box_color, 3)
        """
                        b = bboxes[i]
                if len(self._heatmap_array.shape) == 2: # greyscale image
                    self._heatmap_array = cv2.cvtColor(self._heatmap_array, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(self._heatmap_array, (b[0], b[1]), (b[2], b[3]),\
                        color, 3)
        """
        cv2.imshow('camera', frame)
        c = cv2.waitKey(10)
        if c == ord('q') or not valid:
            break

    cap.release()
    cv2.destroyAllWindows()