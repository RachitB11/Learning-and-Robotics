Project 3 ESE 650: 2D SLAM of THOR Humanoid using LIDAR and IMU Data

See rachitbh_2DSLAM.pdf for details on methodology.

Run test.py to run the algorithm. LIDAR Data and RGB data is available on request

DEPENDENCIES:
matplotlib
numpy
os
cv2
math
random 
copy 
pickle
mpl_toolkits

FILES AND DIRECTORIES:

test.py: Runs the 2D SLAM algorithm

helper_func.py: Contains helper functions for the algorithm like Importance resampling, Initializing MAP, Plotting Data etc.

particle.py: Contains the particle class and all the methods related to it. (Transformation to world, finding map correlation, prediction step etc.)

Video Results: Contains the side by side comparision of the raw RGB data vs the map created by the algorithm. Used to evaluate performance of the algorithm