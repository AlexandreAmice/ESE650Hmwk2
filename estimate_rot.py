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
	dataNum = 3
	imuVals = scipy.io.loadmat("imu/imuRaw" + str(dataNum) + ".mat")['vals']
	data = ImuToPhysical(imuVals)
	imuts = scipy.io.loadmat("imu/imuRaw" + str(dataNum) + ".mat")['ts']
	P = 20 * np.eye(6)
	R = 100 * np.eye(6)  # estQ(data[0:100])
	Q = 0.25* np.eye(6)
	roll, pitch, yaw = ukf(data, imuts, P, R, Q)
	return roll,pitch,yaw



