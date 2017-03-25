Project 1 ESE 650: Color Segmentation using GMMs

NOTE: The project is done on easy dataset.

Run test_script.py to run the algorithm. There is one sample image in the folder as an example.

DEPENDENCIES:
matplotlib
numpy
os
cv2
math
random 
copy 
pickle

FILES AND DIRECTORIES:

test_script.py : Runs the algorithm iteratively over the images. Add the test folder name to the current directory and rename
folder variable or add the images to the Imade_data director

barrel.py: Contains the barrel_finder function to apply the GMMs and the image processing techniques to extract the barrel.

train.py: Trains the models using the pixel data in the Data folder.

gmm.py: Contains my implementation of the EM algorithm to estimate the parameters of the GMM.

make_data.py: Implements roipoly to extract pixel intensity values for the different classes. (Change folder name as required)

file_reader.py: Concatenates pixel intensity data from different images into one file. (Change folder name as required)

obj : Directory to store the trained models

Results: Directory to store results

Data: Directory to store intensity value