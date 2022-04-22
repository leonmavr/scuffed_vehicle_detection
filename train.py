#!/usr/bin/env python3

from sklearn import svm
import pickle as pkl
import glob
import os
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)

X = []
Y = []

training_path = os.path.join(this_script_folder, 'training_data')

# positive labels
positive_features = glob.glob(os.path.join(training_path, 'vehicles', '*.pkl'))
for f in positive_features:
    f_abs_path = os.path.join(training_path, 'vehicles', f)
    with open(f_abs_path, 'rb') as f:
        xi = pkl.load(f)
    X.append(xi)
    Y.append(1)

# negative labels
negative_features = glob.glob(os.path.join(training_path, 'non-vehicles', '*.pkl'))
for f in negative_features:
    f_abs_path = os.path.join(training_path, 'non-vehicles', f)
    with open(f_abs_path, 'rb') as f:
        xi = pkl.load(f)
    X.append(xi)
    Y.append(0)

clf = svm.SVC()
clf.fit(X, Y)
with open('car_classifier.pkl', 'wb') as f:
    pkl.dump(clf, f)
print('SVM training done. Exported the binary classifier as car_classifier.pkl')
