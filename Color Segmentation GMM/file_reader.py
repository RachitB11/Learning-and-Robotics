import os
import numpy as np

# Open file containing training data
folder = "Proj1_Train_easy"
ite = 0
for filename in os.listdir(folder):
    ite += 1
    print str(ite) + ' ' + filename
    # Open the file, load the data and store the data in a numpy array
    # Concatenate with the big array
    if os.path.isfile(r"Data\\"[:-1] + filename + '_red_dat.txt'):
        with open(r"Data\\"[:-1] + filename + '_red_dat.txt', 'r') as f:
            if ite == 1:
                R_dat = np.loadtxt(f)
            else:
                R_dat = np.concatenate((R_dat, np.loadtxt(f)), axis=0)
    if os.path.isfile(r"Data\\"[:-1] + filename + '_yellow_dat.txt'):
        with open(r"Data\\"[:-1] + filename + '_yellow_dat.txt', 'r') as f:

            if ite == 1:
                Y_dat = np.loadtxt(f)
            else:
                Y_dat = np.concatenate((Y_dat, np.loadtxt(f)), axis=0)
    if os.path.isfile(r"Data\\"[:-1] + filename + '_black_dat.txt'):
        with open(r"Data\\"[:-1] + filename + '_black_dat.txt', 'r') as f:
            if ite == 1:
                BLA_dat = np.loadtxt(f)
            else:
                BLA_dat = np.concatenate((BLA_dat, np.loadtxt(f)), axis=0)
    if os.path.isfile(r"Data\\"[:-1] + filename + '_brown_dat.txt'):
        with open(r"Data\\"[:-1] + filename + '_brown_dat.txt', 'r') as f:
            if ite == 1:
                BR_dat = np.loadtxt(f)
            else:
                BR_dat = np.concatenate((BR_dat, np.loadtxt(f)), axis=0)
    # print filename
    # if os.path.isfile(r"Data\\"[:-1] + filename + '_ored_dat.txt'):
    #     with open(r"Data\\"[:-1] + filename + '_ored_dat.txt', 'r') as f:
    #         print f
    #         if ite == 1:
    #             OR_dat = np.loadtxt(f)
    #             print OR_dat.dtype
    #         else:
    #
    #             print np.loadtxt(f).dtype
    #             OR_dat = np.vstack((OR_dat, np.loadtxt(f)))
    # Just to change the number of files used for the model (Finally comment it to take all the files)
    # if ite >= 35:
    #     break
# Store the cumulative data in a file
with open(r"Data\\"[:-1] + 'red_dat.txt', 'w') as f:
    np.savetxt(f, R_dat, fmt='%.2f')
with open(r"Data\\"[:-1] + 'yellow_dat.txt', 'w') as f:
    np.savetxt(f, Y_dat, fmt='%.2f')
with open(r"Data\\"[:-1] + 'black_dat.txt', 'w') as f:
    np.savetxt(f, BLA_dat, fmt='%.2f')
with open(r"Data\\"[:-1] + 'brown_dat.txt', 'w') as f:
    np.savetxt(f, BR_dat, fmt='%.2f')
# with open(r"Data\\"[:-1] + 'ored_dat.txt', 'w') as f:
#     np.savetxt(f, OR_dat, fmt='%.2f')




