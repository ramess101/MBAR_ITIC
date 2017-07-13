from __future__ import division
import numpy as np 
import os.path

iEpsRerun = int(np.loadtxt('iEps_iteration'))
X0 = np.loadtxt('X0_current')
X1 = np.loadtxt('X1_current')
X2 = np.loadtxt('X2_current')
X3 = np.loadtxt('X3_current')
TOL_MBAR = np.loadtxt('TOL_MBAR')

R_ratio=0.61803399
C_ratio = 1-R_ratio

if iEpsRerun > 1:
    
    F2 = np.loadtxt('F2_current')
    F1 = np.loadtxt('F1_current')

    f = open('F1_previous','w')
    f.write(str(F1))
    f.close()

    f = open('F2_previous','w')
    f.write(str(F2))
    f.close()

    if F2 < F1:

        X0 = X1
        X1 = X2  
        X2 = R_ratio * X1 + C_ratio * X3
        F1 = F2
        eps_it = X2

    else:

        X3 = X2
        X2 = X1
        X1 = R_ratio * X2 + C_ratio * X0
        F2 = F1
        eps_it = X1

    f = open('X0_current','w')
    f.write(str(X0))
    f.close()

    f = open('X1_current','w')
    f.write(str(X1))
    f.close()

    f = open('X2_current','w')
    f.write(str(X2))
    f.close()

    f = open('X3_current','w')
    f.write(str(X3))
    f.close()

    f = open('F1_current','w')
    f.write(str(F1))
    f.close()

    f = open('F2_current','w')
    f.write(str(F2))
    f.close()

    f = open('eps_it_'+str(iEpsRerun),'w')
    f.write(str(eps_it))
    f.close()


#else:

    #eps_guess = np.loadtxt('eps_guess')

    #if iEpsRerun == 0:

    #    eps_it = eps_guess
    
    #elif iEpsRerun == 1:

    #    X_interior = np.array([X1 X2])

    #    eps_it = X_interior[X_interior <> eps_guess]

    #f = open('eps_it_'+str(iEpsRerun),'w')
    #f.write(str(eps_it))
    #f.close()


if np.abs(X3-X0) > TOL_MBAR*(np.abs(X1)+np.abs(X2)):
    conv_MBAR=0
else:
    conv_MBAR=1

f = open('conv_MBAR','w')
f.write(str(conv_MBAR))
f.close()