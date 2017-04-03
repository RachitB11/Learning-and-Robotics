import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import numpy as np
import os
from roipoly import roipoly

# Folder containing images
folder = "Image_data"
for filename in os.listdir(folder):
    # create image
    img = mpimg.imread(os.path.join(folder, filename))
    print filename
    print img.shape
    # Show image and ask user to make a polygon for the 5 different classes
    pl.imshow(img)
    pl.title("Mark Barrel Right Click When Done")
    ROI_red = roipoly(roicolor='r')
    pl.imshow(img)
    pl.title("Mark Yellow Patch Right Click When Done")
    ROI_yellow = roipoly(roicolor='y')
    pl.imshow(img)
    pl.title("Mark Black Right Click When Done")
    ROI_black = roipoly(roicolor='k')
    pl.imshow(img)
    pl.title("Mark Brown Patch Right Click When Done")
    ROI_brown = roipoly(roicolor='m')
    pl.imshow(img)
    pl.title("Mark Other Red Right Click When Done")
    ROI_ored = roipoly(roicolor='r')
    # If a polygon is made store the points inside the polygon using a binary mask and the original image
    # Save the data in a text file
    # Stores the data in RGB format. MatplotLib stores in BG. OpenCV requires RGB
    if ROI_red.allxpoints:
        img_maskR = ROI_red.getMask(img[:, :, 0])
        R_dat = img[img_maskR, ::-1] * 255
        with open(r"Data\\"[:-1]+filename+'_red_dat.txt', 'a') as f:
            np.savetxt(f, R_dat, fmt='%.2f')
    if ROI_yellow.allxpoints:
        img_maskY = ROI_yellow.getMask(img[:, :, 0])
        Y_dat = img[img_maskY, ::-1] * 255
        with open(r"Data\\"[:-1]+filename+'_yellow_dat.txt', 'a') as f:
            np.savetxt(f, Y_dat, fmt='%.2f')
    if ROI_black.allxpoints:
        img_maskBLA = ROI_black.getMask(img[:, :, 0])
        BLA_dat = img[img_maskBLA, ::-1] * 255
        with open(r"Data\\"[:-1]+filename+'_black_dat.txt', 'a') as f:
            np.savetxt(f, BLA_dat, fmt='%.2f')
    if ROI_brown.allxpoints:
        img_maskBR = ROI_brown.getMask(img[:, :, 0])
        BR_dat = img[img_maskBR, ::-1] * 255
        with open(r"Data\\"[:-1]+filename+'_brown_dat.txt', 'a') as f:
            np.savetxt(f, BR_dat, fmt='%.2f')
    if ROI_ored.allxpoints:
        img_maskOR = ROI_ored.getMask(img[:, :, 0])
        OR_dat = img[img_maskOR, ::-1] * 255
        with open(r"Data\\"[:-1]+filename+'_ored_dat.txt', 'a') as f:
            np.savetxt(f, OR_dat, fmt='%.2f')
