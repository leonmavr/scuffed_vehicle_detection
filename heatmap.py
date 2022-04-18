#!/usr/bin/env python3

from cv2 import cv2
import numpy as np
from typing import List, Tuple


class Heatmap(object):
    """docstring for Heatmap"""
    def __init__(self, threshold=3):
        self._threshold = threshold
        self._heatmap_array = None

    def _heatmap(self, frame: np.ndarray,
        bboxes: List[Tuple[int,int,int,int]],
        dbg=False):
        """heatmap.

        Parameters
        ----------
        frame : np.ndarray
            The frame in which objects are detected. We need to know this to
            establish the size (width x height) of the heatmap
        bboxes : List[Tuple(int,int,int,int)]
            A list of bounding boxes to update the heat map with
        threshold :
            Any pixel with values less than this in self._heatmap_array will
            be set to 0
        dbg :
            Show debugging information?
        """
        self._heatmap_array = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        for box in bboxes:
            x0, y0, x1, y1 = box
            self._heatmap_array[y0:y1, x0:x1] += 1
        self._heatmap_array[self._heatmap_array < self._threshold] = 0
        self._heatmap_array[self._heatmap_array >= self._threshold] = 255
        if dbg:
            cv2.imshow('dbg', self._heatmap_array)
            cv2.moveWindow('dbg', 50, 50)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()

    def refine(self, dbg=False) -> List[Tuple[int, int, int, int]]:
        """refine. Assigns one bounding box to each polygon in the heat map.
        Therefore needs the `_heatmap` to be called prior to it. It's a way
        of doing non-maximum suppression.

        Parameters
        ----------
        dbg :
            Show debugging information?

        Returns
        -------
        List[Tuple[int, int, int, int]]
            A list of bounding boxes are (x0, y0, x1, y1)
        """
        # find external contours of the heatmap polygons
        contours, _ = cv2.findContours(self._heatmap_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # The bounding boxes around each heat map polygon. To be returned
        bounding_boxes = [(None, None, None, None)]*len(contours)
        color = (50, 255, 0)
        for i, cnt in enumerate(contours):
            # Find the equivalent area of a SOLID rectangle with the same area as
            # the polygon inside the contour,
            # then find the bounding box of the equivalent rectangle.
            area = cv2.contourArea(cnt)
            x, y, wbig, hbig = cv2.boundingRect(cnt)
            extent = float(area) / (wbig*hbig)
            cx, cy = x + wbig/2, y + hbig/2
            wsmall, hsmall = wbig*extent, np.sqrt(extent)*hbig
            bounding_boxes[i] = (int(cx - wsmall/2), int(cy - hsmall/2),\
                int(cx + wsmall/2), int(cy + hsmall/2))
            # draw debugging information - contours, bounding boxes, etc.
            if dbg:
                b = bounding_boxes[i]
                if len(self._heatmap_array.shape) == 2: # greyscale image
                    self._heatmap_array = cv2.cvtColor(self._heatmap_array, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(self._heatmap_array, (b[0], b[1]), (b[2], b[3]),\
                        color, 3)
        # show the bounding boxes for debugging
        if dbg:
            cv2.imshow('dbg', self._heatmap_array)
            cv2.moveWindow('dbg', 50, 50)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        return bounding_boxes
