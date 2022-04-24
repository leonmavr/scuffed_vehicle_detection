#!/usr/bin/env python3

from cv2 import cv2
import numpy as np
from typing import List, Tuple
import doctest


class Heatmap(object):
    """
    Heatmap class
    Registers bounding boxes and performs non-maximum suppression on them via the
    `_register_boxes` and `refine` methods respectively.

    >>> bboxes = [\
        (100, 200, 200, 300),\
        (125, 225, 225, 325),\
        (150, 250, 250, 350),\
        (175, 275, 200, 300),\
        (300, 400, 400, 500),\
        (350, 450, 400, 500),\
        (500, 50, 75, 75),\
        (425, 400, 450, 425),\
        (425, 400, 450, 425),\
        (425, 400, 450, 425)\
    ]
    >>> shape = (700, 600)
    >>> hm = Heatmap(shape, dbg=False)
    >>> len(hm.refine(bboxes)) == 2
    True
    """
    def __init__(self,
                 shape: Tuple[int, int],
                 threshold: int = 2,
                 dbg: bool = False):
        """
        Parameters
        ----------
        shape: Tuple[int, int]
            The shape of the heatmap array as (rows, columns)
        threshold :
            Any pixel with values less than this in self._heatmap_array will
            be set to 0 during the refinement process
        dbg :
            Show debugging information/figures?
        """
        self._threshold = threshold
        self._shape = shape
        self._dbg = dbg
        # an unrefined 2D array with same shape as ._heatmap array for debugging
        self._heatmap_array_dbg = None
        # the heatmap array to be processed and refined
        self._heatmap_array = None
        # initialises heatmap array(s) according to _shape
        self.reset()

    def reset(self):
        self._heatmap_array = np.zeros(self._shape, np.uint8)
        if self._dbg:
            self._heatmap_array_dbg = np.zeros(self._shape, np.uint8)

    def _register_boxes(self,
        bboxes: List[Tuple[int,int,int,int]]):
        """_heatmap.

        Parameters
        ----------
        frame : np.ndarray
            The frame in which objects are detected. We need to know this to
            establish the size (width x height) of the heatmap
        bboxes : List[Tuple(int,int,int,int)]
            A list of bounding boxes to update the heat map with
        """
        for box in bboxes:
            x0, y0, x1, y1 = box
            self._heatmap_array[y0:y1, x0:x1] += 1
        if self._dbg:
            leq_than_thresh = np.bitwise_and(self._heatmap_array <= self._threshold, self._heatmap_array > 0)
            gt_than_thresh = self._heatmap_array > self._threshold
            self._heatmap_array_dbg = self._heatmap_array.copy()
            self._heatmap_array_dbg[leq_than_thresh] = 100
            self._heatmap_array_dbg[gt_than_thresh] = 255
        self._heatmap_array[self._heatmap_array <= self._threshold] = 0
        self._heatmap_array[self._heatmap_array > self._threshold] = 255

    def refine(self,
               bboxes: List[Tuple[int, int, int, int]],)\
            -> List[Tuple[int, int, int, int]]:
        """refine. Assigns one bounding box to each polygon in the heat map
        that's passed the threshold test in ._register_boxes method.

        Parameters
        ----------
        bboxes : List[Tuple(int,int,int,int)]
            A list of bounding boxes to update the heat map with

        Returns
        -------
        List[Tuple[int, int, int, int]]
            The bounding boxes around the  polygons derived by refining the bounding box inputs
            as a list of (x0, y0, x1, y1) coordinates
        """
        self._register_boxes(bboxes)
        # list of bounding boxes to return
        ret = []
        # find external contours of the heatmap polygons
        contours, _ = cv2.findContours(self._heatmap_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_color = (50, 255, 0)
        for i, cnt in enumerate(contours):
            # Find the equivalent area of a SOLID rectangle with the same area as
            # the polygon inside the contour,
            # then find the bounding box of the equivalent rectangle.
            area = cv2.contourArea(cnt)
            x, y, wbig, hbig = cv2.boundingRect(cnt)
            extent = float(area) / (wbig*hbig)
            cx, cy = x + wbig/2, y + hbig/2
            wsmall, hsmall = wbig*extent, np.sqrt(extent)*hbig
            ret.append((int(cx - wsmall/2), int(cy - hsmall/2),
                int(cx + wsmall/2), int(cy + hsmall/2)))
            # draw debugging information - contours, bounding boxes, etc.
            if self._dbg:
                b = ret[-1]
                if len(self._heatmap_array_dbg.shape) == 2: # greyscale image
                    self._heatmap_array_dbg = cv2.cvtColor(self._heatmap_array_dbg, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(self._heatmap_array_dbg, (b[0], b[1]), (b[2], b[3]),
                                contour_color, 3)
        # show the bounding boxes for debugging
        if self._dbg:
            cv2.imshow('dbg', self._heatmap_array_dbg)
            cv2.moveWindow('dbg', 50, 50)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        return ret