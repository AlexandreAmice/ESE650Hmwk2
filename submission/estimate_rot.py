#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import numpy as np
import scipy.io
from KalmanFilters import *
import quat

def estimate_rot(data_num=1):
	imuVals = scipy.io.loadmat("imu/imuRaw" + str(data_num) + ".mat")['vals']
	data = ImuToPhysical(imuVals)
	imuts = scipy.io.loadmat("imu/imuRaw" + str(data_num) + ".mat")['ts']
	P = 20 * np.eye(6)
	off1A = 0.001
	off2A = 0.0025
	off1G = 0.001
	off2G = 0.0025
	base = np.array([[1, off1A, off2A, 0, 0,   0],
					[ off1A, 1,   off1A,  0, 0,   0],
					[ off2A, off1A,   1,    0, 0,   0],
					[ 0, 0,   0,    1, off1G, off2G],
					[ 0, 0,   0,    off1G, 1, off1G],
					[ 0, 0,   0,    off2G, off1G,   1]])
	R = 75*base   # estQ(data[0:100])
	Q = 0.25*np.eye(6) #base
	roll, pitch, yaw = ukf(data, imuts, P, R, Q)
	return roll,pitch,yaw



