import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.cluster import KMeans
import math
import cv2

# Functions to load save the trained models
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# def train_HMM(gest, folder, N, M):
kmeans = load_obj('kmeans_mod')
# In alphabetical order
gest = ['beat3','beat4','circle','eight','inf','wave']
N = 15
M = 30
scal = 1e-10
train_fol = "tot_Data"

# np.seterr(all='ignore')
mods = load_obj('models')

correct = 0.
total = 0.

# for m in xrange(len(mods)):
#     im =cv2.resize(mods[m]['A'],None,fx=30,fy=30)
#     cv2.imshow("A",1.*(im>1e-1))
#     im2 = cv2.resize(mods[m]['B'], None, fx=10, fy=10)
#     cv2.imshow("B", 1. * (im2 > 1e-1))
#     cv2.waitKey(0)




for g in gest:
    print g
    z_list = []
    for fi in os.listdir(train_fol):
        if fi.startswith(g):
            f_name = os.path.join(train_fol, fi)
            arr = np.loadtxt(f_name)
            dat = arr[:, 1:]
            z = kmeans.predict(dat)
            T = z.shape[0]
            # print len(mods)
            log_lik = []
            for m in xrange(len(mods)):
                print m
                A = mods[m]['A']
                B = mods[m]['B']
                Pi = mods[m]['Pi']

                # Forward algo to calculate alphas
                alp0 = Pi * B[z[0], :]
                # print alp0
                alp0 /= max([scal,np.sum(alp0)])
                alp = np.zeros((T,N))
                alp[0,:] = alp0.reshape(1, -1)

                c_list = [1./max([scal,np.sum(alp0)])]
                ct_list = [1./max([scal,np.sum(alp0)])]

                for i in xrange(T - 1):
                    new_alp = B[z[i + 1], :] * np.dot(A,alp[i, :].reshape(-1, 1)).T
                    # print np.sum(new_alp)
                    c = 1./max([scal,np.sum(new_alp)])

                    ct = 1./max([scal,np.sum(new_alp)])
                    # print c
                    new_alp *= c
                    # print np.sum(new_alp)
                    c_list.append(c)
                    ct_list.append(ct)
                    alp[i+1,:] = new_alp.reshape(-1)
                l = -np.sum(np.log(np.asarray(ct_list)))
                # if math.isnan(l):
                #     log_lik.append(-float("inf"))
                # else:
                log_lik.append(l)
            # print log_lik
            idx = log_lik.index(max(log_lik))
            # print log_lik
            if gest[idx] == g:
                correct += 1.
            total += 1.

print correct/total

