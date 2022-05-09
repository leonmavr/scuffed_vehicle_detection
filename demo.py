#!/usr/env/bin python3

from hog import hog
from heatmap import Heatmap
from cv2 import cv2
import pickle
import os
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def load_classifier(fpath = os.path.join(this_script_folder, 'models',
                    'car_classifier_kaggle_64x64_grey.pkl')):
    with open(fpath, 'rb') as f:
        clf = pickle.load(f)
    return clf


if __name__ == '__main__':
    ### modify this section to change the demo properties
    # strip of image starting from the top where we do attempt to perform detections
    horizon = 0.6
    assert 0 <= horizon <= 1, "The horizon lies from 0 to 1 relative to the bottom of the frame"
    # pre-trained binary classifier for vehicle detection
    clf = load_classifier()
    # video data
    vid = os.path.join(this_script_folder, 'test_data', 'videos', 'udacity.mp4')
    # roughly the size of the object in the video
    vehicle_height = 160
    vehicle_width = vehicle_height
    coarseness = int(vehicle_height/2.0)
    # any region with hits (counts) less than this threshold will not be detected
    heatmap_threshold = 0
    bounding_box_color = (50, 255, 0) # as BGR
    downscale = 4.0

    ### do not modify anything below
    if not 0.99 < downscale < 1.01:
        vehicle_width = int(vehicle_width / downscale)
        vehicle_height = int(vehicle_height / downscale)
        coarseness = int(vehicle_height / 3.0)
    cap = cv2.VideoCapture(vid)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    heatmap = None

    i = 0
    while True:
        valid, frame = cap.read()
        if not valid:
            break
        if heatmap is None:
            heatmap = Heatmap((frame.shape[0], frame.shape[1]), heatmap_threshold)
        # the HOG feature detector expects a 2D matrix
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows, cols = frame.shape[0], frame.shape[1]
        if not 0.99 < downscale < 1.01:
            # opencv references matrices as (x, y) and numpy as (rows, columns)
            frame = cv2.resize(frame, (int(cols/downscale), int(rows/downscale)))

        horizon = int((1-horizon)*rows)
        # bounding boxes to be refined if they contain a positive detection
        bboxes = []
        for y in range(0, rows-vehicle_height, coarseness):
            for x in range(0, cols-vehicle_width, coarseness):
                if y > horizon:
                    #cv2.line(frame, (0,y), (cols, y), (50, 50, 255), 3)
                    roi = frame[y:y+vehicle_height, x:x+vehicle_width]
                    roi = frame[x:x + vehicle_width, y:y+vehicle_height]
                    if roi.shape[0] != 64 and roi.shape[1] != 64:
                        roi = cv2.resize(roi, (64, 64))
                    if clf.predict([hog(roi, rows=64, cols=64)])[0] == 1:
                        cv2.imwrite('det_%03d.png' % i, roi)
                        #cv2.rectangle(frame, (x, y), (x+vehicle_width, y+vehicle_height), (0, 255, 0), 3)
                        i += 1
                        bboxes.append((x, x+vehicle_width, y, y+vehicle_height))
                        if len(bboxes) >= 3:
                            import pdb; pdb.set_trace()
        bboxes = heatmap.refine(bboxes)
        cv2.imwrite("hm_%03d.jpg" % i, heatmap._heatmap_array)

        heatmap.reset()
        for b in bboxes:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), bounding_box_color, 3)
        cv2.imshow('detections', frame)
        c = cv2.waitKey(10)
        cv2.imwrite('out_%03d.png' % i, frame)
        i += 1
        if c == ord('q') or not valid:
            break

    cap.release()
    cv2.destroyAllWindows()