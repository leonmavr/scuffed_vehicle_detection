import numpy as np
from cv2 import cv2
from typing import List, Union
import doctest

def update_histogram(grad_magn: float,
        grad_angle: float,
        angle_bins: List,
        votes: Union[List, None] = None) -> List[float]:
    """update_histogram. Updates histogram of magnitudes (aka 'votes') vs
    angles for Histogram of Oriented Gradients (HOG). This is accomplished
    by spreading the gradient magnitude `grad_magn` (vote) across the `votes`
    vector according to the 2 nearest bins of the angle `grad_angle` on the
    `angle_bins` vector. A basic doctest is defined below.
    Parameters
    ----------
    grad_magn : float
        The gradient magnitude as computed by the image gradient 
    grad_angle : float
        The gradient angle (degrees) as computed by the image gradient 
    angle_bins : List
        A list of angles the represent the votes vs angles histogram bins.
        Assuming they're uniform.
    votes : Union[List, None]
        The votes vector. It represents the count, or weight, across each
        histogram bin.
    >>> angle_h = [10, 90, 170]
    >>> votes = [0, 0, 0]
    >>> votes = update_histogram(100, 8, angle_h, votes)
    >>> [int(v) for v in votes] == [100, 0, 0]
    True
    >>> votes = update_histogram(100, 172, angle_h, votes)
    >>> [int(v) for v in votes] == [100, 0, 100]
    True
    >>> votes = update_histogram(100, 150, angle_h, votes)
    >>> [int(v) for v in votes] == [100, 25, 175]
    True
    Returns
    -------
    List[float]
        An updated vector of histogram votes
    """
    if votes is None:
        votes = [0.0 for _ in angle_bins]
    grad_angle %= 180.0
    # assume uniformly-spaced bins on x-axis
    bin_width = angle_bins[1] - angle_bins[0]
    # prepare to 'spread' gradient magnitude across 2 nearest bins
    bin0 = angle_bins[0]
    ind_binl, ind_binr = \
        int(max(0, (grad_angle-bin0)//bin_width)),\
        min(int((grad_angle-bin0)//bin_width+1), len(angle_bins)-1)
    bin1, bin2 = angle_bins[ind_binl], angle_bins[ind_binr]
    # bilinear interpolation across 2 nearest bins
    votes[ind_binl] += grad_magn + grad_magn*(bin1 - grad_angle)/bin_width -\
        int(grad_angle < angle_bins[0])*grad_magn
    votes[ind_binr] += grad_magn + grad_magn*(grad_angle - bin2)/bin_width -\
        int(grad_angle > angle_bins[-1])*grad_magn
    return votes


def hog(impath: str, rows = 128, cols = 64, stride = 8):
    """hog. Computes the histogram of gradient of an image as a vector.
    This vector can later be combined with Support Vector Machines to
    create an object detection pipeline. Implementation reference:
    https://www.youtube.com/watch?v=0Zib1YEE4LU
    Parameters
    ----------
    impath : str
        File path to image to read from
    rows :
        Number of rows of the resized image containing the object of interest
    cols :
        Number of columns of the resized image containing the OOI
    stride :
        How many pixels to iterate by before every histogram of orientations
        is computed
    """
    ### Step 1 - resize image
    im = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (cols, rows))
    ### Step 2 - compute gradient magnitude and angle
    gradx = cv2.Sobel(np.array(im, np.float32)/255, cv2.CV_32F, dx = 1, dy = 0, ksize = 1)
    grady = cv2.Sobel(np.array(im, np.float32)/255, cv2.CV_32F, dx = 0, dy = 1, ksize = 1)
    grad_magn, grad_angle = cv2.cartToPolar(gradx, grady, angleInDegrees = True)
    ### Step 3 - compute orientation histogram
    # the centre of the bins of the angle histogram
    # used for angle quantisation of each 16x16 block
    angle_bins = list(range(10, 180, 20))
    hist_matrix = np.zeros((cols//stride, rows//stride), dtype = np.ndarray)
    # scan the resized imagr row by row in 8x8 blocks, compute their histograms
    for iy, y in enumerate(range(0, rows, stride)):
        for ix, x in enumerate(range(0, cols, stride)):
            patch_angles = np.array(grad_angle[x:x+stride, y:y+stride], np.float32).ravel()
            patch_magns = np.array(grad_magn[x:x+stride, y:y+stride]).ravel()
            votes = [0 for _ in angle_bins]
            for a, m in zip(patch_angles, patch_magns):
                update_histogram(m, a, angle_bins, votes)
            hist_matrix[ix, iy] = votes
    ### Step 4 - 16x16 block normalisation
    # collects the features in one vector after block normalisation
    feature_vec = np.array([], np.float32)
    # scan the histogram matrix in 2x2 squares and compute the concatenated
    # histogram of all histograms under each 2x2 square
    for iy1, iy2 in zip(range(hist_matrix.shape[1]-1), range(1, hist_matrix.shape[1])):
        for ix1, ix2 in zip(range(hist_matrix.shape[0]-1), range(1, hist_matrix.shape[0])):
            # flatten all elements in the 4 8x8 cells
            flat = np.array(sum(hist_matrix[ix1:ix2+1, iy1:iy2+1].ravel(), []), np.float32)
            # normalise the flattened result
            norm_l2 = np.linalg.norm(flat)
            near_zero = 1e-28
            if norm_l2 > near_zero:
                flat /= norm_l2
            for f in flat:
                feature_vec = np.append(feature_vec, f)
    # in the classical paper, vector below will have 7x15x9x4 = 3780 features
    return feature_vec 
