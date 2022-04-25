#!/usr/bin/env python3

### Tip:
# For more refined but more computationally expensive feature vectors,
# you can change the `cols` and `rows` parameters in `hog` to higher
# values, e.g. rows=128, cols=128.
# Row and column aspect ratio depends on the object you're aiming to
# detect, so for vehicles it's recommended that they're equal.
###

import os
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
import sys
sys.path.append(os.path.join(this_script_folder, '..'))
from hog import hog
import pickle as pkl
from sklearn import svm

training_dir = os.path.join(this_script_folder, 'kaggle')

# SVM training vectors
X, Y = [], []
positive_files = os.listdir(os.path.join(training_dir, 'vehicles'))
negative_files = os.listdir(os.path.join(training_dir, 'non-vehicles'))
n_pos = len(positive_files)
n_neg = len(negative_files)

for i, img_file in enumerate(positive_files):
    try:
        f_abs_path = os.path.join(training_dir, 'vehicles', img_file)
        feature_vec = hog(f_abs_path, rows=64, cols=64, mode = 'grey')
        X.append(feature_vec)
        Y.append(1)
        if i % 1000 == 0:
            print("positive training:", int(100*i/n_pos), "%")
    except:
        pass


for i, img_file in enumerate(negative_files):
    try:
        f_abs_path = os.path.join(training_dir, 'non-vehicles', img_file)
        feature_vec = hog(f_abs_path, rows=64, cols=64, mode = 'grey')
        X.append(feature_vec)
        Y.append(0)
        if i % 1000 == 0:
            print("negative training:", int(100*i/n_neg), "%")
    except:
        pass

clf = svm.SVC()
clf.fit(X, Y)
with open('car_classifier_kaggle_64x64_grey.pkl', 'wb') as f:
    pkl.dump(clf, f)
print('SVM training done. Exported the binary classifier as car_classifier*.pkl')
