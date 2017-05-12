import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.cluster import KMeans
import math

N = 15
M = 30
train_fol = "tot_Data"
gest = ['beat3','beat4','circle','eight','inf','wave']
scal = 1e-10
iter_max = 100
eps = 1e-7
plot_cost = False

# Functions to load and store the trained models
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def init_params(N, M):

    # A = np.random.rand(N, N)
    # A = A/np.sum(A,axis=0)
    # B = np.ones((M, N))
    # B = B/np.sum(B,axis=0)
    # Pi = np.random.rand(N)
    # Pi /= np.sum(Pi)

    A_i = np.eye(N) + np.eye(N, k=-1)
    A_i[0,-1] = 1.
    A_i *= 0.5

    B_i = np.ones((M,N))/float(M)

    Pi_i = np.zeros(N)
    Pi_i[0] = 1.

    return A_i, B_i, Pi_i

i = 1
for fi in os.listdir(train_fol):
    print fi
    if fi.endswith(".txt"):
        f_name = os.path.join(train_fol, fi)
        arr = np.loadtxt(f_name)
        if i == 1:
            dat = arr[:, 1:]
        else:
            dat = np.vstack((dat, arr[:, 1:]))
        i += 1
kmeans = KMeans(n_clusters=M, init='k-means++', max_iter=100).fit(dat)

save_obj(kmeans, 'kmeans_mod')
kmeans = load_obj('kmeans_mod')


# def train_HMM(gest, folder, N, M):

train_lab = kmeans.labels_
cum_T = 0
mods = []
for g in gest:
    print g
    z_list = []
    for fi in os.listdir(train_fol):
        if fi.startswith(g):
            f_name = os.path.join(train_fol, fi)
            arr = np.loadtxt(f_name)
            T = arr.shape[0]
            z = train_lab[cum_T: (cum_T + T - 1)]
            cum_T += T
            z_list.append(z)
    A, B, Pi = init_params(N, M)
    old_cost = float("inf")
    current_cost = 0.0
    cost_list = []

    # while math.fabs(current_norm - old_norm) > eps:
    for loops in xrange(iter_max):
        print loops
        alp_tot = []
        bet_tot = []
        gamma_tot = []
        Xi_tot = []

        # z_list = [z_list[0]]

        for z in z_list:
            T = z.shape[0]
            # Forward algo to calculate alphas
            alp0 = Pi * B[z[0], :]
            alp0 /= np.sum(alp0)
            alp = np.zeros((T,N))
            alp[0,:] = alp0.reshape(1, -1)

            c_list = [1/np.sum(alp0)]
            ct_list = [1/np.sum(alp0)]

            for i in xrange(T - 1):
                new_alp = B[z[i + 1], :] * np.dot(A,alp[i, :].reshape(-1, 1)).T
                # print np.sum(new_alp)
                c = 1./max([scal,np.sum(new_alp)])
                ct = 1./np.sum(new_alp)
                # print c
                new_alp *= c
                # print np.sum(new_alp)
                c_list.append(c)
                ct_list.append(ct)
                alp[i+1,:] = new_alp.reshape(-1)

            # Backward algo to calculate betas
            betT = np.ones(N) * ct_list[T-1]
            bet = np.zeros((T, N))
            bet[-1,:] = betT

            for i in xrange(T-2,-1,-1):
                new_bet = np.dot(B[z[i+1], :], (A*bet[i+1, :].reshape(-1,1)))
                new_bet *= c_list[i]
                bet[i,:] = new_bet.reshape(-1)

            # Calculate gamma
            gamma = ((alp*bet).T / np.sum(alp*bet,axis=1)).T
            # print np.sum(gamma,axis =  1)

            # Calculate Xi
            Xi = np.zeros((N,N,T-1))

            for i in xrange(T-1):
                Xi[:, :, i] = (A*alp[i,:])*((B[z[i+1], :]*bet[i+1,:]).reshape(-1,1))
                Xi[:, :, i] /= np.sum(Xi[:, :, i])

            alp_tot.append(alp)
            bet_tot.append(bet)
            gamma_tot.append(gamma)
            Xi_tot.append(Xi)

        # Update parameters

        K = len(alp_tot)

        # Update pi
        temp = np.zeros(N)
        for i in xrange(K):
            temp += gamma_tot[i][0,:]
        Pi = temp / K
        # Pi[Pi<1e-10] = 1e-10
        # print np.sum(Pi)

        # Update A
        temp_A = np.zeros((N,N))
        temp_norm = np.zeros(N)
        for i in xrange(K):
            temp_A += np.sum(Xi_tot[i], axis=2)
            temp_norm += np.sum(gamma_tot[i][:-1,:], axis=0)

        # print temp_norm
        A = temp_A / temp_norm
        # A[A<1e-10] = 1e-10
        # print np.sum(A,axis=0)

        # Update B
        temp_norm = np.zeros(N)
        temp_B = np.zeros((M,N))
        cent = np.arange(M).reshape(-1,1)
        for i in xrange(K):
            z = z_list[i]
            labs = np.matlib.repmat(cent, 1, z.shape[0])
            z_arr = np.matlib.repmat(z, M, 1)
            mask = 1.*np.equal(z_arr,labs)
            # print mask.shape
            # print gamma_tot[i].shape
            temp_B += np.dot(mask, gamma_tot[i])
            temp_norm += np.sum(gamma_tot[i], axis=0)
        B = temp_B / temp_norm
        # B[B<1e-10] = 1e-10
        # print np.sum(B,axis=0)

        current_cost = -np.sum(np.log(np.asarray(ct_list)))
        print (current_cost-old_cost)
        if math.fabs(current_cost-old_cost)<eps:
            print (current_cost - old_cost)
            break
        else:
            old_cost = current_cost
        cost_list.append(current_cost)

    if plot_cost:
        plt.plot(cost_list)
        plt.ylabel('Log Likelihood')
        plt.show()


    cur_mod = {}
    # A[A <= 1e-10] = 0.
    # B[B <= 1e-10] = 0.
    # Pi[Pi <= 1e-10] = 0.

    cur_mod['A'] = A
    cur_mod['B'] = B
    cur_mod['Pi'] = Pi
    print cur_mod
    mods.append(cur_mod)

save_obj(mods,'models')