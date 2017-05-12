import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.cluster import KMeans
import math

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
test_fol = "test_multiple"
save_images = True

# np.seterr(all='ignore')
mods = load_obj('models')

correct = 0.
total = 0.

case = 0
for fi in os.listdir(test_fol):
    f_name = os.path.join(test_fol, fi)
    arr = np.loadtxt(f_name)
    dat = arr[:, 1:]
    z = kmeans.predict(dat)
    T = z.shape[0]
    # print len(mods)
    log_lik = []
    for m in xrange(len(mods)):
        # print m
        A = mods[m]['A']
        B = mods[m]['B']
        Pi = mods[m]['Pi']

        # Forward algo to calculate alphas
        alp0 = Pi * B[z[0], :]
        # print alp0
        alp0 /= max([scal,np.sum(alp0)])
        alp = np.zeros((T,N))
        alp[0,:] = alp0.reshape(1, -1)

        c_list = [1. / max([scal, np.sum(alp0)])]
        ct_list = [1. / max([scal, np.sum(alp0)])]

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
        if math.isnan(l):
            log_lik.append(-float("inf"))
        else:
            log_lik.append(l)
    # print log_lik
    idx = log_lik.index(max(log_lik))
    conf = (1./np.asarray(log_lik))/np.sum(1./np.asarray(log_lik))
    # print conf
    print gest[idx]
    if save_images:
        x = np.arange(len(gest))
        plt.bar(x, height=list(conf))
        plt.xticks(x + .5, gest)
        plt.xlabel('Gestures')
        plt.ylabel('Confidence')
        case += 1
        s = 'im' + str(case) + '.png'
        plt.savefig(s, bbox_inches='tight')
        plt.clf()


