import numpy as np
import numpy.linalg as npl
import scipy.io
import scipy as sp
from functools import partial
from math import cos, sin, exp, pi
from quat import *
import matplotlib.pyplot as plt

def ukf_update_q(stateVect, P0, obsVect, R, Q, dt):
    '''
    #data sheet for observation is at https://d1b10bmlvqabco.cloudfront.net/attach/jqzm87i1qhb68t/is9oo9a6f3c1gb/js6kaguyn3m1/IMU_reference.pdf
    :param stateVect: passed as [position quaternion, angular velocity] = [q0 qx qy qz wx wy wz]
    :param P0: Covariance of state vector. Should be 6x6
    :param obsVect: passed as [accelerometer data, gyro data] = [ax, ay, az, wz, wx, wy] (this is not a typo the axis for the two measurements are not the same]
    :param R: covariance of observation noise
    :param Q: covariance of state noise
    :param dt: time step
    :return:
    '''
    stateVect = stateVect.astype(float)
    accData = obsVect[0:3]
    gyroData = np.array([obsVect[4], obsVect[5], obsVect[3]])  # rectify observation data to fit with state vector convention
    obsVect = np.concatenate((accData, gyroData), 0)
    q0 = stateVect[0:4]
    (n,_) =  P0.shape
    omega0 = stateVect[4:]
    S = np.transpose(npl.cholesky(P0 + Q))

    #create Sigma points
    #print "start sigma point creation"
    # Wi = np.concatenate(( sqrt(2 * n) * S, -sqrt(2 * n) * S), 1)
    k = 0.5
    alpha = 8
    #Wi = np.concatenate(( sqrt( k*n+alpha) * S, -sqrt(k*n+alpha) * S), 1)
    # Adjust the first column here to add the mean vector
    Wi = np.concatenate((np.zeros((n,1)), sqrt(k*n+alpha)*S, -sqrt(k*n+alpha)*S),1)

    Xi = np.concatenate((np.zeros((1,Wi.shape[1])), np.zeros_like(Wi)))
    for c in range(Wi.shape[1]):
        posNoise = Wi[0:3,c]
        velNoise = Wi[3:6,c]
        qw = unitQuat(posNoise)
        posVect = quatMult(q0, qw)
        velVect = omega0 + velNoise
        Xi[:,c] = np.concatenate((posVect, velVect))

    #apply state transformation
    #print "doing state transformation"
    actionQuat = unitQuat(omega0*dt)
    Yi = np.zeros_like(Xi)
    for c in range(Yi.shape[1]):
        qi = quatMult(Xi[0:4,c], actionQuat)
        omegai = Xi[4:7,c]
        Yi[:,c] = np.concatenate((qi,omegai))

    x0bar = np.zeros_like(stateVect)
    x0bar[4:] = np.mean(Yi[4:,:],1)
    qbar = quatMult(stateVect[0:4],actionQuat)
    #mean quat vector update
    theShape = Yi.shape[1]
    #when averaging you have a bunch of 0 terms is this a problem/
    #print "finding mean vector"
    qbar, eiVect = quatAveraging(qbar, Yi[0:4,:])
    Wiprime = np.zeros((6,Yi.shape[1]))
    x0bar[0:4] = qbar


    for c in range(theShape):
        Wiprime[0:3, c] = eiVect[:,c]
        Wiprime[3:6, c] = Yi[4:7, c] - x0bar[4:7]
    P0bar = 1/(2.0*n) *np.matmul(Wiprime, np.transpose(Wiprime))
    # temp = np.concatenate((x0bar[0:4], gyroData))
    # return temp, P0bar

    #apply observer transformation
    #print "applying observer transformation"
    Zi = np.zeros((6,Xi.shape[1])) #each observation is passed as [Ax, Ay, Az, Wz, Wx, Wy]
    gvect = np.array([0,0,0, 9.89])
    for c in range(Zi.shape[1]):
        qi = Xi[0:4,c]
        #what is my g vector at the current position?
        Zi[0:3,c] = quatToVect(quatMult(quatInv(qi),quatMult(gvect, qi, False), False))
        Zi[3:7,c] = Xi[4:7,c]

    z0bar = np.mean(Zi,1)
    ZiNoMean = (Zi.transpose() - z0bar).transpose()
    Pzz = 1/(2.0*n) *np.matmul(ZiNoMean, ZiNoMean.transpose())

    v0 = obsVect-z0bar
    Pvv = Pzz + R

    Pxz = 1/(2.0*n) * np.matmul(Wiprime, np.transpose(ZiNoMean))

    #kalman gain
    K = np.matmul(Pxz, npl.inv(Pvv))

    newState = np.zeros_like(stateVect)
    stateUpdate = np.matmul(K,v0)
    updateQuat = unitQuat(stateUpdate[0:3])
    newState[0:4] = quatMult(q0, updateQuat)
    newState[4:7] = omega0 + stateUpdate[3:6]
    newCov = P0bar - np.matmul(K, np.matmul(Pvv, K.transpose()))
    return newState, newCov

def ukf(observations, timesteps, P0, R, Q):
    t0 = time.time()
    initquat = unitQuat(observations[0:3,0])
    initAng = observations[3:6,0]
    initState = np.concatenate((initquat, initAng)) #np.array([1,1,0,0, 1,.5,.5])
    numItems = min(observations.shape[1]-1, timesteps.shape[1]-1) #800
    roll, pitch, yaw = np.zeros(numItems),np.zeros(numItems),np.zeros(numItems)
    curState = initState
    curCov = P0
    for j in range(0,numItems):
        curObs = observations[:,j]
        dt = timesteps[0,j+1]-timesteps[0,j]
        curState, curCov = ukf_update_q(curState, curCov, curObs, R, Q, dt)
        rotMat = quatToRot(curState)
        roll[j], pitch[j], yaw[j] = rotToRollPitchYaw(rotMat)
    t1 = time.time()
    return roll, pitch, yaw


def ImuToPhysical(data):
    data = data.astype('float')
    quant = 1023.0
    vrefA = 3200 #mv
    vrefG = 3300 #mv
    biasA = 1600 #mv
    biasG = 1230 #mv
    sensitivityA = 295 #mv/g
    sensitivityG = 3.33 #mv/deg/sec
    g = 9.8 #m/s^2
    valA = lambda raw: ((raw*vrefA)/quant-biasA)/sensitivityA*g
    valG = lambda raw: ((raw*vrefG)/quant-biasG)/sensitivityG*pi/180
    # bias = np.array([370.2, 374, 375.7])
    # scaleFact = 0.0162
    # valG = lambda raw: (raw-bias)*scaleFact
    for i in range(0,data.shape[1]):
        data[0:3,i] = valA(data[0:3,i])
        data[0:2,i] = -data[0:2,i] #Ax and Ay are backwards
        data[3:6,i] = valG(data[3:6,i])
    return data

def ImuToRollPitch(data, raw = True):
    '''
    :param data: imu data
    :param raw: set to false if IMU data has already been converted to real data
    :return:
    '''
    if raw:
        data = ImuToPhysical(data)
    numData = data.shape[1]
    roll, pitch = np.zeros(numData), np.zeros(numData)
    for c in range(0,numData):
        aX, aY, aZ = data[0,c], data[1,c], data[2,c]
        pitch[c] = atan2(aX , sqrt(aY**2 + aZ**2))
        roll[c] = atan2(aY, sqrt(aX**2 + aZ**2))
    return roll,pitch

def estQ(data):
    '''
    pass as real data. only pass a small subset
    :param data:
    :return:
    '''
    meanData = np.mean(data,1)
    dataNoMean = (data.transpose()-meanData).transpose()
    Q = np.matmul(dataNoMean,dataNoMean.transpose())
    return Q

def rmse(predictions, targets):
    maxIdx = min(len(predictions), len(targets))
    return np.sqrt(((predictions[:maxIdx] - targets[:maxIdx]) ** 2).mean())



if __name__ == '__main__':
    dataNum = 3
    imu1Vals = scipy.io.loadmat("imu/imuRaw" + str(dataNum) + ".mat")['vals']
    data = ImuToPhysical(imu1Vals)
    imu1ts = scipy.io.loadmat("imu/imuRaw" + str(dataNum) + ".mat")['ts']
    viconMats = scipy.io.loadmat("vicon/viconRot" + str(dataNum) + ".mat")['rots']
    viconTs = scipy.io.loadmat("vicon/viconRot" + str(dataNum) + ".mat")['ts']
    P = 20*np.eye(6)
    R = 0.025*np.eye(6) #estQ(data[0:100])
    Q = 10*np.eye(6)
    roll, pitch, yaw = ukf(data, imu1ts, P, R, Q)


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
