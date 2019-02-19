#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import numpy as np
import scipy.io
from KalmanFilters import *
from scipy.signal import filtfilt, lfilter
from scipy import signal
import quat

def estimate_rot(data_num=1):
    print "Amice submission"
    imuVals = scipy.io.loadmat("imu/imuRaw" + str(data_num) + ".mat")['vals']
    data = ImuToPhysical(imuVals)
    imuts = scipy.io.loadmat("imu/imuRaw" + str(data_num) + ".mat")['ts']
    P = np.eye(6)
    off1A = 0.025
    off2A = 1
    off1G = 0.06
    off2G = 1
    base = np.array([[1, off1A, off2A, 0, 0,   0],
                    [ off1A, 1,   off1A,  0, 0,   0],
                    [ off2A, off1A,   1,    0, 0,   0],
                    [ 0, 0,   0,    1, off1G, off2G],
                    [ 0, 0,   0,    off1G, 1, off1G],
                    [ 0, 0,   0,    off2G, off1G,   1]])
    base = np.matmul(base.transpose(),base)+10**-5*np.eye(6)
    R = 150*np.eye(6)   # np.eye(6)#estQ(data[0:100])
    Q = 0.09*np.eye(6) #np.eye(6) # np.eye(6) #base
    roll, pitch, yaw = ukf(data, imuts, P, R, Q)
    return roll, pitch, yaw

def low_pass(roll, pitch, yaw):
    # [b, a] = signal.butter(4, 0.008, btype = 'lowpass', analog=False)
    # roll = filtfilt(b,a, roll)
    # pitch = filtfilt(b, a, pitch)
    # yaw = filtfilt(b, a, yaw)
    for i in range(1,len(roll)):
        thresh = 0.1
        trust = 0.99
        if abs(roll[i]-roll[i-1]) > thresh:
            roll[i] = (1-trust) *roll[i] + trust * roll[i-1]
        if abs(pitch[i]-pitch[i-1]) > thresh:
            pitch[i] = (1-trust) *pitch[i] + trust * pitch[i-1]
        if abs(yaw[i]-yaw[i-1]) > thresh:
            yaw[i] = (1-trust) *yaw[i] + trust * yaw[i-1]

    return roll, pitch, yaw



