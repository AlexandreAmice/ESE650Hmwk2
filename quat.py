import numpy as np
import scipy as sp
import scipy.linalg as spl
from math import cos,atan2,sin,acos,asin,sqrt
import numpy.linalg as npl
import sys
from functools import partial
import time

def quatFromComps(scalar, vector, unit = True):
    '''
    takes a scalar unit and a vector and outputs a 4x1 numpy array as a column
    :param scalar:
    :param vector:
    :param unit: boolean to determine whether to return a unit quaternion or not
    :return:
    '''
    scalar = float(scalar)
    vector = vector.astype(float)
    if unit:
        norm = scalar**2 + np.matmul(np.transpose(vector), vector)
    else:
        norm = 1
    return np.reshape(np.insert(vector, 0, [scalar]), (4,))/norm

def unitQuat(vect):
    a = npl.norm(vect)
    if a > 10**-10:
        e = vect/a
        return quatFromComps(cos(a/2.0), sin(a/2.0)*e)
    else:
        return np.array([1,0,0,0])

def quatFromOmegaTheta(theta, omega):
    scalar = cos(theta/2.0)
    vect = sin(theta/2.0)*omega
    return quatFromComps(scalar, vect)

def vectToZeroQuat(vect):
    return np.concatenate((np.array([0]), vect))

def quatFromAngle(theta, omega):
    '''
    make a quaternion from an angle and rotational rate
    :param theta: angle
    :param omega: unit vector of direction of axis
    :return: quaternion as column
    '''
    scalar = cos(theta/2.0)
    s = sin(theta/2.0)
    return quatFromComps(scalar, s*omega)

def quatMult(quat1, quat2, normalize = True):
    '''
    multiply two quaternions. Coded by hand using hamiltonian formula due to speed
    :param quat1:
    :param quat2:
    :return:
    '''
    a1, b1, c1, d1 = quat1
    a2, b2, c2, d2 = quat2
    ret = np.array([a1*a2-b1*b2-c1*c2-d1*d2,
                     a1*b2+b1*a2+c1*d2-d1*c2,
                     a1*c2-b1*d2+c1*a2+d1*b2,
                     a1*d2 + b1*c2 - c1*b2 +d1*a2] , dtype = np.float64)
    if normalize:
        return ret/npl.norm(ret)
    else:
        return ret





def quatToRot(quat):
    '''
    change a quaternion to its matrix rotation
    :param quat:
    :return:
    '''
    u0 = quat[0]
    u = quat[1:4]
    uHat = vectToCross(u)
    u = np.reshape(u, (3,1))
    H =  (u0**2 - np.matmul(np.transpose(u), u))*np.eye(3) + 2*u0*uHat + 2*np.matmul(u, np.transpose(u))
    return np.reshape(H, (3,3))

def quatNorm(quat):
    return quat[0]**2+np.matmul(quat[1:3],quat[1:3].transpose())

def quatToVect(quat):
    if abs(quat[0]) > 1:
        print quat
        print quatNorm(quat)
    theta = 2*np.arccos(quat[0])
    if theta < 10**-10:
        return quat[1:4]
    scale = sin(theta/2.0)
    return quat[1:4]/scale



def vectToCross(vect):
    '''
    map a 3x1 vector to its cross product matrix
    :param vect:
    :return:
    '''
    return np.array([[0, -vect[2], vect[1]],
                     [vect[2], 0, -vect[0]],
                     [-vect[1], vect[0], 0]])

def crossMatToVect(omega):
    return np.array([omega[2,1], omega[0,2], omega[1,0]])

def quatInv(quat):
    '''
    conjugate a quaternion
    :param quat:
    :return:
    '''
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])
    u0 = quat[0]
    u = quat[1:4]
    return np.insert(-u, 0, u0)


def rotToQuat(R):
    theta = cos((np.trace(R) - 1)/2.0)
    omegaHat = (R- R.transpose())/(2*sin(theta))
    vect = crossMatToVect(omegaHat)
    return quatToVect(vect)

def rotToThetaOmega(R):
    theta = acos((np.trace(R) - 1) / 2.0)
    omegaHat = (R - R.transpose()) / (2 * sin(theta))
    vect = crossMatToVect(omegaHat)
    return theta, vect

def thetaOmegaToRot(theta, omega):
    omegaHat = vectToCross(omega)
    R = np.eye(3)+sin(theta)*omegaHat + (1-cos(theta)) *npl.matrix_power(omegaHat,2)
    return R

def rotToRollPitchYaw(R):
    yaw = atan2(R[1,0],R[0,0])
    pitch = atan2(-R[2,0],sqrt(R[2,1]**2+R[2,2]**2))
    roll = atan2(R[2,1],R[2,2])
    return roll,pitch, yaw

def isUnitQuat(quat):
    value = sqrt(np.matmul(quat, quat.transpose()))
    tolerance = 10**(-6)
    return value > 1-tolerance and value < 1 + tolerance

def quatAveraging(initQuat, quatSet):
    '''
    :param quatSet: passed as a matrix with columns being unit quats
    :return: average quaternion and final error vectors
    '''
    numQuats = quatSet.shape[1]
    t = 0
    qtbar = initQuat
    eiQuat = np.zeros_like(quatSet)
    eiVect = np.zeros((quatSet.shape[0]-1,numQuats))
    e = sys.maxint * np.ones((3,1))
    while t < 10**6 and npl.norm(e) > 10**-2:
        for i in range(numQuats):
            qi = quatSet[:,i]
            eiQuat[:,i] = quatMult(qi,quatInv(qtbar))
            eiVect[:,i] = quatToVect(eiQuat[:,i]).transpose()
        e = 1/(2.0*numQuats) * np.sum(eiVect, 1)
        equatTot = unitQuat(e)
        qtbar = quatMult(equatTot, qtbar)
        t +=1
    return qtbar, eiVect

def quatToRollPitchYaw(quat):
    temp = quatToRot(quat)
    return rotToRollPitchYaw(temp)

if __name__ == '__main__':
    R = np.eye(3)
    theta, omega = rotToThetaOmega(R)
    print omega






