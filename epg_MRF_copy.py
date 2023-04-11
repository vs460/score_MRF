#!/usr/bin/python

# EPG CPMG simulation code, based off of Matlab scripts from Brian Hargreaves <bah@stanford.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>

import numpy as np
from warnings import warn

def rf(FpFmZ, alpha):
    """ Propagate EPG states through an RF rotation of 
    alpha (radians). Assumes CPMG condition, i.e.
    magnetization lies on the real x axis.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians

    OUTPUT:
        FpFmZ = Updated FpFmZ state.
        RR = RF rotation matrix (3x3).

    """

    # -- From Weigel at al, JMRI 41(2015)266-295, Eq. 21.

    
    if abs(alpha) > 2 * np.pi:
        warn('rf2: Flip angle should be in radians! alpha=%f' % alpha)

    cosa2 = np.cos(alpha/2.)**2
    sina2 = np.sin(alpha/2.)**2

    cosa = np.cos(alpha)
    sina = np.sin(alpha)

    RR = np.array([ [cosa2, sina2, sina],
                    [sina2, cosa2, -sina],
                    [-0.5 * sina, 0.5 * sina, cosa] ])


    FpFmZ = np.dot(RR, FpFmZ)

    return FpFmZ

def rf_ex(FpFmZ, alpha):
    """ Propagate EPG states through an RF excitation of 
    alpha (radians) along the x direction, i.e. phase of 0.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians

    OUTPUT:
        FpFmZ = Updated FpFmZ state.
        RR = RF rotation matrix (3x3).

    """

    try:
        alpha = alpha[0]
    except:
        pass


    if abs(alpha) > 2 * np.pi:
        warn('rf2_ex: Flip angle should be in radians! alpha=%f' % alpha)

    cosa2 = np.cos(alpha/2.)**2
    sina2 = np.sin(alpha/2.)**2

    cosa = np.cos(alpha)
    sina = np.sin(alpha)

    RR = np.array([ [cosa2, sina2, -1j*sina],
                    [sina2, cosa2, 1j*sina],
                    [-0.5j * sina, 0.5j * sina, cosa] ])

    FpFmZ = np.dot(RR, FpFmZ)

    return FpFmZ

def relax_mat(T, T1, T2):
    E2 = np.exp(-T/T2)
    E1 = np.exp(-T/T1)

    EE = np.diag([E2, E2, E1])      # Decay of states due to relaxation alone.

    return EE

def relax(FpFmZ, T, T1, T2):
    """ Propagate EPG states through a period of relaxation over
    an interval T.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        T1, T2 = Relaxation times (same as T)
        T = Time interval (same as T1,T2)

    OUTPUT:
        FpFmZ = updated F+, F- and Z states.
        EE = decay matrix, 3x3 = diag([E2 E2 E1]);

   """

    E2 = np.exp(-T/T2)
    E1 = np.exp(-T/T1)

    EE = np.diag([E2, E2, E1])      # Decay of states due to relaxation alone.
    RR = 1 - E1                     # Mz Recovery, affects only Z0 state, as 
                                    # recovered magnetization is not dephased.


    FpFmZ = np.dot(EE, FpFmZ)       # Apply Relaxation
    FpFmZ[2,0] = FpFmZ[2,0] + RR    # Recovery  

    return FpFmZ



def grad(FpFmZ, noadd=False):
    """Propagate EPG states through a "unit" gradient. Assumes CPMG condition,
    i.e. all states are real-valued.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        noadd = True to NOT add any higher-order states - assume
                that they just go to zero.  Be careful - this
                speeds up simulations, but may compromise accuracy!

    OUTPUT:
        Updated FpFmZ state.

    """

    # Gradient does not affect the Z states.

    if noadd == False:
        FpFmZ = np.hstack((FpFmZ, [[0],[0],[0]]))   # add higher dephased state

    FpFmZ[0,:] = np.roll(FpFmZ[0,:], 1)     # shift Fp states
    FpFmZ[1,:] = np.roll(FpFmZ[1,:], -1)    # shift Fm states
    FpFmZ[1,-1] = 0                         # Zero highest Fm state
    FpFmZ[0,0] = FpFmZ[1,0]                 # Fill in lowest Fp state

    return FpFmZ



def MRF_TE(FpFmZ, alpha, TE, TD, T1, T2, noadd=False, recovery=True):
    """ Propagate EPG states through a full TE, i.e.
    relax -> grad -> rf -> grad -> relax.
    Assumes CPMG condition, i.e. all states are real-valued.

    INPUT:
        FpFmZ = 3xN vector of F+, F- and Z states.
        alpha = RF pulse flip angle in radians
        T1, T2 = Relaxation times (same as TE)
        TE = Echo Time interval (same as T1, T2)
        noadd = True to NOT add any higher-order states - assume
                that they just go to zero.  Be careful - this
                speeds up simulations, but may compromise accuracy!

    OUTPUT:
        FpFmZ = updated F+, F- and Z states.

   """    

    EE = relax_mat(TE/2., T1, T2)

    # FpFmZ = grad(FpFmZ, noadd)
    FpFmZ = rf_ex(FpFmZ, alpha)
    # FpFmZ = grad(FpFmZ, noadd)
    if recovery:
        FpFmZ = relax(FpFmZ, TE, T1, T2)
    else:
        FpFmZ = np.dot(EE, FpFmZ)
    
    FpFmZ_echo = FpFmZ
    if recovery:
        FpFmZ = relax(FpFmZ, TD, T1, T2)
    else:
        FpFmZ = np.dot(EE, FpFmZ)

    return FpFmZ, FpFmZ_echo


def MRF_signal(angles_rad, TR0, T1, T2, B1=1.):
    """Simulate Fast Spin-Echo CPMG sequence with specific flip angle train.
    Prior to the flip angle train, an excitation pulse of angle_ex_rad degrees
    is applied in the Y direction. The flip angle train is then applied in the X direction.

    INPUT:
        angles_rad = array of flip angles in radians equal to echo train length
        TE = echo time/spacing
        T1 = T1 value in seconds
        T2 = T2 value in seconds

    OUTPUT:
        Mxy = Transverse magnetization at each echo time
        Mz = Longitudinal magnetization at each echo time
        
    """

    T = len(angles_rad)
    Mxy = np.zeros((T,1),dtype=complex)
    Mz = np.zeros((T,1),dtype=complex)

    P = np.array([[0],[0],[1]]) # initially on Mz

    try:
        B1 = B1[0]
    except:
        pass

    # pre-scale by B1 homogeneity
    angles_rad = B1 * np.copy(angles_rad)

    TD = np.zeros(len(angles_rad))
    TE = TD.copy();
    sin_theta = np.abs(np.sin(np.cumsum(angles_rad*np.round(np.cos((np.arange(0,len(angles_rad)))*np.pi)))))
    for ip in range(1,len(angles_rad)):
        TD[ip-1] = TR0/2;
        TE[ip]   = TD[ip-1] * sin_theta[ip-1] / sin_theta[ip]
        
        if TE[ip] > TR0/2:
            TE[ip]   = TR0/2;
            TD[ip-1] = TE[ip] * sin_theta[ip] / sin_theta[ip-1]
     
    P[2,0] = -1
    for i in range(T):
        alpha = angles_rad[i]
        P, Signal = MRF_TE(P, np.round(np.cos(i*np.pi))*alpha, TE[i], TD[i], T1, T2)

        Mxy[i] = Signal[0,0]
        Mz[i] = Signal[2,0]

    return Mxy, Mz, TE, TD



    

