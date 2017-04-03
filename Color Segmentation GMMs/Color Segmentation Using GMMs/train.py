import numpy as np
from gmm import expect_max
import pickle

# Function to find unique points in the data and remove repeating data points
def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])

# Functions to load and store the trained models
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Load cumulative RGB intensity data and find the unique data points  for each class
with open(r"Data\\"[:-1] + 'red_dat.txt', 'rb') as f:
    R_dat = np.loadtxt(f,dtype=np.float32)
    print R_dat.shape
    R_dat = unique_rows(R_dat)
    print R_dat.shape
    print '1'
with open(r"Data\\"[:-1] + 'yellow_dat.txt', 'rb') as f:
    Y_dat = np.loadtxt(f, dtype=np.float32)
    print Y_dat.shape
    Y_dat = unique_rows(Y_dat)
    print Y_dat.shape
    print '2'
with open(r"Data\\"[:-1] + 'black_dat.txt', 'rb') as f:
    BLA_dat = np.loadtxt(f, dtype=np.float32)
    print BLA_dat.shape
    BLA_dat = unique_rows(BLA_dat)
    print BLA_dat.shape
    print '3'
with open(r"Data\\"[:-1] + 'brown_dat.txt', 'rb') as f:
    BR_dat = np.loadtxt(f, dtype=np.float32)
    print BR_dat.shape
    BR_dat = unique_rows(BR_dat)
    print BR_dat.shape
    print '4'
with open(r"Data\\"[:-1] + 'ored_dat.txt', 'rb') as f:
    OR_dat = np.loadtxt(f, dtype=np.float32)
    print OR_dat.shape
    OR_dat = unique_rows(OR_dat)
    print OR_dat.shape
    print '5'

# Sampling frequency for passing training data to the EM algorithm
num_sample = 30
# Number of clusters to be trained
num_clust = 3
# Implement the EM algortihm for each class and store the model
mod_R = expect_max(R_dat[::num_sample,:], num_clust, eps=0.001)
save_obj(mod_R,'mod_R_I')
print '6'
mod_Y = expect_max(Y_dat[::num_sample,:], num_clust, eps=0.001)
save_obj(mod_Y,'mod_Y_I')
print '7'
mod_BLA = expect_max(BLA_dat[::num_sample,:], num_clust, eps=0.001)
save_obj(mod_BLA,'mod_BLA_I')
print '8'
mod_BR = expect_max(BR_dat[::num_sample,:], num_clust, eps=0.001)
save_obj(mod_BR,'mod_BR_I')
print '9'
mod_OR = expect_max(OR_dat[::num_sample,:], num_clust, eps=0.001)
save_obj(mod_OR,'mod_OR_I')
print '10'


