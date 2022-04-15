#!/usr/bin/env python3

from sklearn import svm
import pickle as pkl
import os
from hog import hog
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

y_true = []
y_pred = []

# binary classifier
with open('training_data/car_classifier.pkl', 'rb') as f:
    clf = pkl.load(f)

# positive labels
for f in os.listdir(os.path.join(this_script_folder,  'test_data', 'positive')):
    f_abs_path = os.path.join(this_script_folder, 'test_data', 'positive', f)
    with open(f_abs_path, 'rb') as f:
        y_pred.append(clf.predict([hog(f_abs_path)])[0])
    y_true.append(1)

# negative labels
for f in os.listdir(os.path.join(this_script_folder,  'test_data', 'negative')):
    f_abs_path = os.path.join(this_script_folder, 'test_data', 'negative', f)
    with open(f_abs_path, 'rb') as f:
        y_pred.append(clf.predict([hog(f_abs_path)])[0])
    y_true.append(0)

f1Score = f1_score(y_true, y_pred, average=None)
accuracy = accuracy_score(y_true, y_pred)
precision = recall_score(y_true, y_pred)
recall = precision_score(y_true, y_pred)
print('='*30)
print('accuracy:', accuracy)
print('recall:', recall)
print('precision:', precision)
print('f1:', f1Score)
print('='*30)
