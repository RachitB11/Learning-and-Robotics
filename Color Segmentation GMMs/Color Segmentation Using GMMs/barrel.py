import numpy as np
import math
import cv2 as cv2
import copy


def barrel_finder(img, mods):
    # Find number of clusters from length of model
    num_clust = len(mods[0])
    # print img.shape
    # Unwrap the image into an n*3 matrix
    dat = np.hstack((img[:,:,0].flatten().reshape(-1,1),img[:,:,1].flatten().reshape(-1,1),img[:,:,2].flatten().reshape(-1,1)))
    # List to store the probability for each pixel for each model
    res = []
    for i in xrange(len(mods)):
        params = mods[i]
        # Calculating the probability for a particular model
        res.append(sum(params[j]['memb']*np.exp(-0.5* np.sum(np.dot(dat-params[j]['mu'],np.linalg.cholesky(np.linalg.inv(params[j]['sigma'])))** 2, axis=1)) / math.sqrt(np.linalg.det(params[j]['sigma'])) for j in xrange(num_clust)))
    # n*(num models) matrix where each row stores probabiity for a pixel for all the models
    probs = np.column_stack((res[k] for k in xrange(len(res))))
    # Find the class for which it has maximum probability
    col = probs.argmax(axis=1)
    # Make a binary image
    bin_im = np.zeros((dat.shape[0], 1))
    # Set those pixels to 255 which are classified as barrel red
    bin_im[np.asarray(col) == 0, :] = 255
    bin_im = np.asarray(bin_im, dtype="uint8")
    img_rest = bin_im.reshape((img.shape[0], img.shape[1]))
    ret, img_rest = cv2.threshold(img_rest, 254, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Segmented Image', img_rest)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    bw = copy.deepcopy(img_rest)
    # Make the kernel for morphological operations
    kernel = np.ones((10, 10), np.uint8)
    # Perform the closing operation of erosion followed by dilation to remove noise
    bw = cv2.erode(bw, kernel, iterations=1)
    bw = cv2.dilate(bw, kernel, iterations=1)
    cv2.imwrite('BW.png', bw)
    cv2.imshow('Result', bw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, 2)

    rem = copy.deepcopy(contours)
    contours = []
    # print contours

    # Remove contours which are relatively small
    for i, cnt in enumerate(rem):
        M = cv2.moments(cnt)
        if M['m00'] > 300:
            contours.append(rem[i])

    # Function to check if 2 contours are in close proximity
    def contours_near(c1, c2):
        row1, row2 = c1.shape[0], c2.shape[0]
        # Run a nested for loop to check if any point on one one contour is close to another point in the other contour
        # If even one set of points is close enough return true else false
        for i in xrange(row1):
            for j in xrange(row2):
                d = np.linalg.norm(c1[i] - c2[j])
                if abs(d) < 70:
                    return True
                elif i == row1 - 1 and j == row2 - 1:
                    return False

    num_contours = len(contours)
    cont_number = np.zeros((num_contours, 1))

    # If no contours were detected return empty lists
    if num_contours ==0:
        return [],[],[]

    # This nested for loop stores clusters all the contours that are close the same contour number in the list cont_number
    # It basically clusters different contours which are close (as defined by the contours_near function)
    for i, cnt1 in enumerate(contours):
        x = i
        if i != num_contours - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = contours_near(cnt1, cnt2)
                if dist == True:
                    val = min(cont_number[i], cont_number[x])
                    cont_number[x] = cont_number[i] = val
                else:
                    if cont_number[x] == cont_number[i]:
                        cont_number[x] = i + 1

    # List to store the bounding box, depth and centroid
    combined = []
    depth = []
    centroid = []
    # Variable to store the number of cluster of contours
    maximum = int(cont_number.max()) + 1

    # Store parameters. fx and fy were calculated experimentally using the training data
    act_width = 40.0
    act_height = 57.0
    fx = 1046.58
    fy = 1119.26

    # Loop through number of cluster of contours
    for i in xrange(maximum):
        pos = np.where(cont_number == i)[0]
        if pos.size != 0:
            # Stack points of all the contours which belong to the same cluster (have the same contour number)
            cont = np.vstack(contours[i] for i in pos)
            # Make a convex hull around the close contours
            hull = cv2.convexHull(cont)

            # epsilon = 0.1 * cv2.arcLength(hull, True)
            # approx = cv2.approxPolyDP(hull, epsilon, True)
            # print approx.shape

            # Find the minimum rectangular area for the convex hull
            rect = cv2.minAreaRect(hull)
            # x, y, w, h = cv2.boundingRect(hull)

            # Put constraint on angle of tilt of the bounding box and the aspect ratio of the bounding box
            # The height must be at least somewhat greater than the width to be even considered as a barrel
            if (abs(rect[2]) > 70 and rect[1][0] < 1.2*rect[1][1]) or (abs(rect[2])<20 and rect[1][1] < 1.2*rect[1][0]):
                # print str(w)+' '+str(h)
                continue
            # Putting an area constraint on the barrel
            if maximum>1:
                if rect[1][0] * rect[1][1] < 2500:
                    continue
            # Extract the 4 points of the minimum area rectangle
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)

            # If it passes through all the filters add the box to the list
            combined.append(box)
            # Find depth using both fx and fy
            depthx = fx/min(rect[1])*act_width
            depthy = fy/max(rect[1])*act_height
            # depth.append((depthy+depthx)/200.0)

            # Pick minimum depth since partial occlusion might alter one of the parameters
            depth.append(min([depthx,depthy])/100.0)
            centroid.append(rect[0])
    # If no contour passed the filters return empty lists
    if not combined:
        return [],[],[]
    return combined, depth, centroid



