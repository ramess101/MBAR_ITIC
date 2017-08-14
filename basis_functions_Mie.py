# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:08:35 2017

@author: ram9
"""

import numpy as np

eps_low = 1.
eps_high = 2.
sig_high = 1.
sig_low = 0.5

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
lam_sim = 12.       
       
for isim in range(2):
    eps_rerun = eps_sim[isim]
    sig_rerun = sig_sim[isim]
    print('Rerun with epsilon = '+str(eps_rerun)+' and sigma = '+str(sig_rerun))       
    Clam[isim] = eps_rerun * sig_rerun ** lam_sim
    C6[isim] = eps_rerun * sig_rerun ** 6.
    U_sim[isim] = 10. * Clam[isim] + 5. * C6[isim] # This will eventually be replaced by a simulation
    Cmatrix[isim,0] = Clam[isim]
    Cmatrix[isim,1] = C6[isim]

rarray = np.linalg.solve(Cmatrix,U_sim)
print(rarray)