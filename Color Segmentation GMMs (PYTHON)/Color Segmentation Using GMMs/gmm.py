import numpy as np
import math
import random
import copy


# Function to implement expectation maximization algorithm
def expect_max(data, num_clust=2, eps=0.001):
    # Find feature length and number of data points
    num_feat = data.shape[1]
    num_data = data.shape[0]
    # limits = []
    # for i in xrange(num_feat):
    #     limits.append((data[:,i].min(), data[:,i].max()))

    # Function to calculate the Gaussian given data point, mean and covariance matrix
    def gaussian(x, mu, sig):
        x_bar = np.matrix(x - mu)
        # print sig
        info_mat = np.linalg.inv(sig)
        prob = math.exp(-(x_bar * info_mat * np.transpose(x_bar))/2.0)/math.sqrt(((2.0 * math.pi) ** (len(x))) * np.linalg.det(sig))
        return prob

    # Function to initialize the parameters for EM algorithm
    def initialze():
        # Assumed equal membership weights
        memb = 1/float(num_clust)
        # Randomly picked points in data as mean of clusters
        mu = data[int(random.uniform(0, num_data)), :]
        x_bar = data-np.tile(np.mean(data, axis=0),(num_data, 1))
        # Estimate diagonal covariance matrix using variance of the R,G and B intensities in the data
        sigma = np.diag((np.linalg.norm(x_bar, axis=0)**2)/num_data)
        # sigma = np.diag([100.,100.,100.])
        # print sigma
        return {'memb':memb,'mu':mu,'sigma':sigma}

    # Initialize the old_cost function as infinity
    old_cost = float("inf")
    current_cost = 0.0
    # List to hold the parameters of the model
    params = []

    ite = 0

    # Initialize the parameters of each cluster
    for i in xrange(num_clust):
        params.append(initialze())

    # print params

    # EM LOOP
    # Continue loop till the cumulative log likelihood cost function does not converge
    while math.fabs(current_cost-old_cost)>eps:
        ite += 1
        # print ite
        if ite>1:
            old_cost = current_cost
        params_new = []
        # Loop thorugh each cluster
        for i in xrange(num_clust):
            # Initialize the variables to store the weighted sums of the mean,weights andcovariance matrix
            mu = 0.0
            memb = 0.0
            sigma = np.zeros((num_feat,num_feat), dtype=np.float32)
            # print params[i]
            # E - STEP
            for j in xrange(num_data):
                # print sum(params[k]['memb']*gaussian(data[j,:],params[k]['mu'],params[k]['sigma']) for k in xrange(num_clust))
                # Find the membership probability for each of the data points for the current cluster
                r = params[i]['memb']*gaussian(data[j,:],params[i]['mu'],params[i]['sigma'])/ sum(
                    params[k]['memb']*gaussian(data[j,:],params[k]['mu'],params[k]['sigma']) for k in
                    xrange(num_clust))
                # print r
                # Sum the weighted mean and the membership probability
                mu += r*data[j,:]
                memb += r
            # print i,': done'
            for j in xrange(num_data):
                r = params[i]['memb'] * gaussian(data[j, :], params[i]['mu'], params[i]['sigma']) / sum(
                    params[k]['memb'] * gaussian(data[j, :], params[k]['mu'], params[k]['sigma']) for k in
                    xrange(num_clust))
                # Sum the weighted covariance matrix (found using outer product for each point)
                sigma += r * np.outer(data[j, :]-(mu/memb),data[j, :]-(mu/memb))
            # M - STEP
            # Use the weighted sums calculated above to calculate the updated value of the parameters
            params_new.append({'memb': memb/float(num_data),'mu': mu/memb, 'sigma': sigma/memb})
        # print params_new

        current_cost = 0.0
        params = copy.deepcopy(params_new)
        # Use the new parameters to calculate the cumulaive log likelihood value
        for i in xrange(num_data):
            temp = 0.0
            for j in xrange(num_clust):
                prob = params[j]['memb'] * gaussian(data[i, :], params[j]['mu'], params[j]['sigma'])
                temp += prob
            current_cost += math.log(temp)
        # Print current log likelihood and the differnce between old and new likelihoods
        print current_cost
        print 'iteation '+ str(ite)+' : '+str((current_cost - old_cost))

    return params











