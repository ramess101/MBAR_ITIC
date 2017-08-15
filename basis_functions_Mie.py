# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:08:35 2017

@author: ram9
"""

import numpy as np

eps_low = 117.4997658
eps_high = 123.377174904
sig_high = 0.37870061174
sig_low = 0.376842136893

eps_sim = np.zeros(2)
sig_sim = np.zeros(2)
Clam = np.zeros(2)
C6 = np.zeros(2)
U_sim = np.zeros(2)
Cmatrix = np.zeros([2,2])

eps_sim[0] = eps_low
eps_sim[1] = eps_high
sig_sim[0] = sig_high
sig_sim[1] = sig_low
lam_sim = 15.0   

U_sim[0] =  -5588.724609375000
U_sim[1] =  -5889.600097656250
       
for isim in range(2):
    eps_rerun = eps_sim[isim]
    sig_rerun = sig_sim[isim]
    print('Rerun with epsilon = '+str(eps_rerun)+' and sigma = '+str(sig_rerun))       
    Clam[isim] = eps_rerun * sig_rerun ** lam_sim
    C6[isim] = eps_rerun * sig_rerun ** 6.
    #U_sim[isim] = 10. * Clam[isim] + 5. * C6[isim] # This will eventually be replaced by a simulation
    Cmatrix[isim,0] = Clam[isim]
    Cmatrix[isim,1] = C6[isim]

rarray = np.linalg.solve(Cmatrix,U_sim)
print(rarray)

eps_new = 123.451800039
sig_new = 0.375635059575

Clam_new = eps_new * sig_new ** lam_sim
C6_new = eps_new * sig_new ** 6.

Cmatrix_new = np.array([Clam_new,C6_new])

U_new = np.linalg.multi_dot([Cmatrix_new,rarray])
print('Predicted internal energy: '+str(U_new))
print('Actual internal energy was -5899.308105468750')

P_sim = np.zeros(2)

P_sim[0] = 1820.654052734375
P_sim[1] = 1313.558959960938
P_new = 962.420043945312
