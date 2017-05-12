import numpy as np
# a = np.array([[1,1,1],[2,2,2], [3,3,3]])
# b = np.array([[1],[2],[3]])
# print a/b
# c = np.array([1,2,3])
# print a/c
# print a[:,0]
# d = np.vstack((a[:,0],a[:,1]))
# print d
# e = np.dot(a[:,0],a)
# print e
# f = a[:,0].reshape(-1,1)
# f = f.reshape(-1)
# print f.shape[0]
# print f
# g = np.array([1,2,3,4])
# h = np.array([1,1,4,4])
# print g==h

i =  np.eye(4) + np.eye(4,k=-1)
j  = i[:,[0,2]]
print j