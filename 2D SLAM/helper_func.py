import numpy as np
import matplotlib.pyplot as plt
import math as m
from particle import particle


# Set Map parameters
def setMAP_params(MAP):
    MAP['res'] = 0.05  # meters
    MAP['xmin'] = -20  # meters
    MAP['ymin'] = -20
    MAP['xmax'] = 30
    MAP['ymax'] = 30
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    # print MAP['sizex']
    # print MAP['sizey']
    MAP['map'] = np.zeros((MAP['sizey'], MAP['sizex']), dtype=np.int32)
    MAP['l_odds'] = np.zeros((MAP['sizey'], MAP['sizex']), dtype=np.float64)  # DATA TYPE: float64
    return MAP


# Set distance parameters
def setDist_params(dist):
    dist['g2com'] = 0.93
    dist['com2h'] = 0.33
    dist['h2lid'] = 0.15
    dist['g2lid'] = 0.93 + 0.33 + 0.15
    return dist


# Plot poses
def plot_poses(poses, t):
    fig = plt.figure(1)
    s1 = fig.add_subplot(311)
    s1.plot(t,poses[:,0], 'r', label="x")
    box = s1.get_position()
    s1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    s1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    s2 = fig.add_subplot(312)
    s2.plot(t,poses[:,1], 'g', label="y")
    box = s2.get_position()
    s2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    s2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    s3 = fig.add_subplot(313)
    s3.plot(t,poses[:,2], 'b', label="theta")
    box = s3.get_position()
    s3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    s3.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Get 3D laser point coordinates in local scan frame
def get_laser_points(scan, angles):

    # Scans
    ranges = np.double(scan)

    # Extract only those scans and corresponding angles which are within a sensible range
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid].reshape(1, -1)
    angles = angles[indValid].reshape(1, -1)
    # print ranges.shape

    # XYZ poistions of the hit points in the 'planar' sensor frame
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    zs = np.zeros(xs.shape)
    ps = np.vstack((xs, ys, zs, np.ones(xs.shape)))

    return ps


# Update the weights using the corellation
def update_w(w_arr, corr):

    # Update weights using logsumexp. It works but causes resampling almost every step
    l = np.log(w_arr) + corr
    l_max = np.amax(l)
    l_sum = l_max + m.log(np.sum(np.exp(l-l_max)))
    l = l - l_sum
    w_up = np.exp(l)


    # Alternate method of estimating new weights by directly using the corellation rather than soft-max
    # Incorrect but efficient
    # l = w_arr * corr
    # w_up = l/np.sum(l)
    return w_up

# Stratified resampling using updated particles
def resample(p_list):
    new_list = []
    u = np.random.uniform(0, 1./float(len(p_list)), 1)[0]
    j = 0
    c = p_list[0].w
    for k in xrange(len(p_list)):

        # Divide the circle into equal parts. Move to the next particle only when b > cumulative weight
        b = (u + ((float(k))/(float(len(p_list)))))

        # print "heres b"
        # print b
        # print u

        if(b>1):
            print b
            quit()

        while (b > c):
            # print c
            j = (j+1)
            c = (c + p_list[j].w)

        new_list.append(particle(w = 1./float(len(p_list)),x = p_list[j].x))

    return new_list
