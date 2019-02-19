import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from quat import *
from KalmanFilters import ImuToPhysical, ImuToRollPitch
from estimate_rot import estimate_rot
dataNum = 3
viconMats = scipy.io.loadmat("vicon/viconRot" + str(dataNum) + ".mat")['rots']
viconTs = scipy.io.loadmat("vicon/viconRot" + str(dataNum) + ".mat")['ts']
roll, pitch, yaw = estimate_rot(dataNum)

numV = viconMats.shape[2]
rollV, pitchV, yawV = np.zeros(numV), np.zeros(numV), np.zeros(numV)
for i in range(0, len(rollV)):
    rollV[i], pitchV[i], yawV[i] = rotToRollPitchYaw(viconMats[:, :, i])

fig = plt.figure()
ax = fig.add_subplot(311)
ax.set_title('roll')
ax.plot(rollV)
ax.plot(roll)

ax = fig.add_subplot(312)
ax.set_title('pitch')
ax.plot(pitchV)
ax.plot(pitch)

ax = fig.add_subplot(313)
ax.set_title('yaw')
ax.plot(yawV)
ax.plot(yaw)
plt.show()




