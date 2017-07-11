from __future__ import division
import numpy as np 
import os.path

iEpsRef = int(np.loadtxt('/home/ram9/Ethane/Gromacs/TraPPEfs/iEpsref'))
iSigmaRef = int(np.loadtxt('/home/ram9/Ethane/Gromacs/TraPPEfs/iSigref'))

X0 = np.loadtxt('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/X0_current')
X1 = np.loadtxt('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/X1_current')
X3 = np.loadtxt('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/X3_current')

R_ratio=0.61803399
C_ratio = 1-R_ratio

if np.abs(X3-X1) > np.abs(X1-X0):
    X1 = X1
    X2 = X1 + C_ratio*(X3-X1)
    
    f = open('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/eps_it_0','w')
    f.write(str(X1))
    f.close()

    f = open('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/eps_it_1','w')
    f.write(str(X2))
    f.close()

else:
    X2 = X1
    X1 = X2 - C_ratio*(X2-X0)

    f = open('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/eps_it_0','w')
    f.write(str(X2))
    f.close()

    f = open('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/eps_it_1','w')
    f.write(str(X1))
    f.close()

f = open('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/X0_current','w')
f.write(str(X0))
f.close()

f = open('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/X1_current','w')
f.write(str(X1))
f.close()

f = open('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/X2_current','w')
f.write(str(X2))
f.close()

f = open('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/X3_current','w')
f.write(str(X3))
f.close()