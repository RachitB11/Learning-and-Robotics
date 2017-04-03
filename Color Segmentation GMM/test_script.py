import os
import cv2 as cv2
import pickle
import barrel


# Function to load the models
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Load the models for the different classes
params_R = load_obj('mod_R')
params_BLA = load_obj('mod_BLA')
params_BR = load_obj('mod_BR')
params_Y = load_obj('mod_Y')
params_OR = load_obj('mod_OR')

mods = [params_R,params_Y,params_BLA,params_BR,params_OR]
ite = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# Folder from where images are taken
folder = "Image_data"

for filename in os.listdir(folder):
    ite += 1
    img = cv2.imread(os.path.join(folder,filename))
    # img = cv2.resize(img,(1200,900))
    # Use the barrel_finder method in barrel.py to return bounding box, distance and centroid of each barrel
    bbox, dist, cent = barrel.barrel_finder(img, mods)

    # Draw bounding box in the image
    cv2.drawContours(img, bbox, -1, (0, 255, 0), 2)

    print 'Image No[' + str(ite) + ']:'

    # Check if no bounding box is found
    if not bbox:
        print 'No Box!!!'
    # If bounding box exists print information about it
    for i in xrange(len(bbox)):
        print 'Box '+str(i+1)+'\nPoint1: ' + str(bbox[i][0]) + '\nPoint2: ' + str(bbox[i][1]) + '\nPoint3: ' + \
              str(bbox[i][2]) + '\nPoint4: ' + str(bbox[i][3]) + '\nDistance in m: ' + str(dist[i])
        loc = (bbox[i][0][0],bbox[i][2][1]-10)
        temp = round(dist[i],2)
        cv2.putText(img, str(temp), loc, font, 1, (0, 255, 0), 2)
        cv2.circle(img, (int(cent[i][0]),int(cent[i][1])), 5, (0, 255, 0), -11)
    # Write result to file and display the result in window
    cv2.imwrite("Results/result_" + filename, img)
    # cv2.imshow('Result',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
