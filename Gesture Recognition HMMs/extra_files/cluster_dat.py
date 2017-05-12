import numpy as np
import matplotlib.pyplot as plt
import math as m
import cv2
import os
import pickle
from sklearn.cluster import KMeans

# Functions to load and store the trained models
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

i = 1
for fi in os.listdir("Data"):
    print fi
    if fi.endswith(".txt"):
        f_name = os.path.join("Data", fi)
        arr = np.loadtxt(f_name)
        if i == 1:
            dat = arr[:, 1:]
        else:
            dat = np.vstack((dat, arr[:, 1:]))
        i += 1

kmeans = KMeans(n_clusters=15, init='k-means++', max_iter=100).fit(dat)

save_obj(kmeans, 'kmeans_mod')
