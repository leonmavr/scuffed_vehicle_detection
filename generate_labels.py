#!/usr/bin/env python3

### Tip:
# For more refined but more computationally expensive feature vectors,
# you can change the `cols` and `rows` parameters in `hog` to higher
# values, e.g. rows=128, cols=128.
# Row and column aspect ration depend on the object you're aiming to
# detect, so for vehicles it's recommended that they're equal.
###

import os
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
import sys
sys.path.append(os.path.join(this_script_folder, '..'))
from hog import hog
import pickle as pkl

training_dir = os.path.join(this_script_folder, 'training_data')

for f in os.listdir(os.path.join(training_dir, 'vehicles')):
    try:
        f_abs_path = os.path.join(training_dir, 'vehicles', f)
        feature_vec= hog(f_abs_path, rows=64, cols=64)
        with open(f_abs_path.split('.png')[0] + '.pkl', 'wb') as f:
            pkl.dump(feature_vec, f)
    except:
        pass


for f in os.listdir(os.path.join(training_dir, 'non-vehicles')):
    f_abs_path = os.path.join(training_dir, 'non-vehicles', f)
    try:
        feature_vec= hog(f_abs_path, rows=64, cols=64)
        with open(f_abs_path.split('.png')[0] + '.pkl', 'wb') as f:
            pkl.dump(feature_vec, f)
    except:
        pass

