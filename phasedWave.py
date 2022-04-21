import numpy as np
def phasedWave(theta,M,k,d):
    return np.transpose(np.sin(k*d*np.sin(theta)*(np.arange(0,M))));