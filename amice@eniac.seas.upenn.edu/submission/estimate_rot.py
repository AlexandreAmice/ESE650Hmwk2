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
	P = np.eye(6)
	Q = 0.5 * np.eye(6)
	R = 0.5 * np.eye(6)
	roll, pitch, yaw = ukf(data, imuts, P, R, Q)
	return roll,pitch,yaw


roll, pitch, yaw = estimate_rot(3)
fig = plt.figure()
ax = fig.add_subplot(311)
ax.set_title('roll')
ax.plot(roll)

ax = fig.add_subplot(312)
ax.set_title('pitch')
ax.plot(pitch)

ax = fig.add_subplot(313)
ax.set_title('yaw')
ax.plot(yaw)
plt.show()


