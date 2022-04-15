#!/bin/bash

### Download and prepare dataset
echo "[*] Downloading and preparing GTI vehicle dataset"
wget https://github.com/leonmavr/assets/raw/master/gti_training_set/non-vehicles.zip
wget https://github.com/leonmavr/assets/raw/master/gti_training_set/vehicles.zip
unzip "non-vehicles.zip" 
unzip vehicles.zip

mkdir -p training_data
mv non-vehicles training_data
mv vehicles training_data

# flatten the directory contents
cd training_data/vehicles
i=0
for f in `find . -name *png`; do
    mv $f $i.png
    i=$[$i+1]
done
rm -rf Far Left MiddleClose Right

# flatten the directory contents
cd ../../training_data/non-vehicles
i=0
for f in `find . -name *png`; do
    mv $f $i.png
    i=$[$i+1]
done
rm -rf Far Left MiddleClose Right
cd -

### Generate ground truth for each image
echo "[*] Generating ground truth for the dataset"
./generate_labels.py

### Train and export classifier as a pickle file
echo "[*] Training and exporting binary classifier"
./train.py
# Now a *.pkl file should have been created at this directory.
# To use it, load it with pickle and call its predict method on your input vector.

### Join together the pre-trained model(s) (since github doesn't allow uploading files larger than 25M) 
echo "[*] Joining together pre-trained classifier file"
cd models/split_files
cat car_classifier_64x64.aa car_classifier_64x64.ab car_classifier_64x64.ac car_classifier_64x64.ad > ../car_classifier_64x64.pkl
cd ..
rm -rf split_files
cd ..
