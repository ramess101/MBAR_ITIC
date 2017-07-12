from __future__ import division
import numpy as np 
import os.path

X0 = np.loadtxt('X0_current')
X1 = np.loadtxt('X1_current')
X3 = np.loadtxt('X3_current')

R_ratio=0.61803399
C_ratio = 1-R_ratio

if np.abs(X3-X1) > np.abs(X1-X0):
    X1 = X1
    X2 = X1 + C_ratio*(X3-X1)
    
    f = open('eps_it_0','w')
    f.write(str(X1))
    f.close()

    f = open('eps_it_1','w')
    f.write(str(X2))
    f.close()

    f = open('swap_initial','w')
    f.write(str(0))
    f.close()

else:
    X2 = X1
    X1 = X2 - C_ratio*(X2-X0)

    f = open('eps_it_0','w')
    f.write(str(X2))
    f.close()

    f = open('eps_it_1','w')
    f.write(str(X1))
    f.close()

    f = open('swap_initial','w')
    f.write(str(1))
    f.close()

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