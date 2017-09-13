# -*- coding: utf-8 -*-
"""
This code takes a single reference, single snapshot, example to help develop the basis function methodology.
The code is not intended to be clean or optimized but is intended to be clear enough for validation purposes.

"""
from __future__ import division
import numpy as np


bonds='LINCS'

nm3tom3 = 1e27
kJm3tobar = 1./100.
NA = 6.022140857e23

#Constant with respect to force field       
KE = 1185.967041015625      
# Constants from state point
Lbox = 3.2168 #[nm]
Vbox = Lbox**3. #[nm3]

# Not needed anymore
#eps_low = 117.4997658
#eps_high = 123.377174904
#sig_high = 0.37870061174
#sig_low = 0.376842136893

eps_basis = np.zeros(2)
sig_basis = np.zeros(2)
Clam = np.zeros(2)
C6 = np.zeros(2)
U_basis = np.zeros(2)
Vir_basis = np.zeros(2)
VirXX_basis = np.zeros(2)
VirYY_basis = np.zeros(2)
VirZZ_basis = np.zeros(2)
Pdc_basis = np.zeros(2)
P_basis = np.zeros(2)
PXX_basis = np.zeros(2)
PYY_basis = np.zeros(2)
PZZ_basis = np.zeros(2)
Cmatrix = np.zeros([2,2])

# Old data

#eps_basis[0] = eps_low
#eps_basis[1] = eps_high
#sig_basis[0] = sig_high
#sig_basis[1] = sig_low
#lam_basis = 15.0
#
#U_basis[0] =  -5588.724609375000
#U_basis[1] =  -5889.600097656250
#
#P_basis[0] = 1820.654052734375
#P_basis[1] = 1313.558959960938

#eps_new = 123.451800039
#sig_new = 0.375635059575
#U_new = -5899.308105468750
#P_new = 962.420043945312

#Taken from LJtoMie_Ufirst_lam13 ref0 Isochore rho0 T0, reruns 1,2

eps_basis[0] = 104.466306711
eps_basis[1] = 116.387441424   
sig_basis[0] = 0.375138117621
sig_basis[1] = 0.375768262189
         
lam_basis = 13.0
N_basis = (lam_basis/(lam_basis-6.))*(lam_basis/6.)**(6./(lam_basis-6.))

#Taken from the first snapshot, where ref0 is TraPPE

if bonds == 'LINCS':

    # Constrained with LINCS
    U_basis[0] =  -5432.041503906250
    U_basis[1] =  -6055.096191406250
           
    #Virial XX,YY,ZZ
    VirXX_basis[0] = 212.644409179688
    VirXX_basis[1] = 80.657455444336
    VirYY_basis[0] = 161.777313232422
    VirYY_basis[1] = 22.583724975586
    VirZZ_basis[0] = -78.521842956543
    VirZZ_basis[1] = -249.451858520508
       
    #Pressure XX,YY,ZZ
    PXX_basis[0] = 190.447357177734
    PXX_basis[1] = 322.132843017578    
    PYY_basis[0] = 216.155670166016  
    PYY_basis[1] = 355.031341552734
    PZZ_basis[0] = 481.431274414062
    PZZ_basis[1] = 651.970825195312 

elif bonds == 'Harmonic':
             
    # Harmonic bonds
    
    U_basis[0] =  -5431.397949
    U_basis[1] =  -6054.361816
             
    #Virial XX,YY,ZZ
    VirXX_basis[0] = 723.5198364
    VirXX_basis[1] = 672.4127808
    VirYY_basis[0] = 646.8673706
    VirYY_basis[1] = 589.6341553
    VirZZ_basis[0] = 279.9454956
    VirZZ_basis[1] = 185.5500641
       
    #Pressure XX,YY,ZZ
    PXX_basis[0] = -319.2722778
    PXX_basis[1] = -268.2819519    
    PYY_basis[0] = -267.854248  
    PYY_basis[1] = -210.7517548
    PZZ_basis[0] = 123.8242874
    PZZ_basis[1] = 218.0041199           
                
       
#Virial and pressure average (with respect to XX,YY,ZZ-not time)
for ibasis in range(2):
    Vir_basis[ibasis] = (VirXX_basis[ibasis]+VirYY_basis[ibasis]+VirZZ_basis[ibasis])/3.
    P_basis[ibasis] = (PXX_basis[ibasis]+PYY_basis[ibasis]+PZZ_basis[ibasis])/3.
       
#Dispersive contributions
Pdc_basis[0] = -127.706733703613
Pdc_basis[1] = -143.719940185547  
         
Virdc_basis = (KE/3-Pdc_basis/2*Vbox/nm3tom3/kJm3tobar*NA)/3.
       
# Values for a different force field, not used in the basis function development
       
eps_new = 101.250841825
sig_new = 0.37511601053

if bonds == 'LINCS':
    
    # Constrained LINCS
    U_new = -5264.714843750000
    VirXX_new = 214.462371826172
    VirYY_new = 166.006591796875
    VirZZ_new = -66.664085388184
    Pdc_new = -123.732154846191
    PXX_new = 188.633544921875
    PYY_new = 211.936065673828
    PZZ_new = 469.600616455078

elif bonds == 'Harmonic':
    
    # Harmonic bonds
    U_new = -5264.100586
    VirXX_new = 709.1787109
    VirYY_new = 633.9317017
    VirZZ_new = 275.1833801
    Pdc_new = -123.732154846191
    PXX_new = -304.9638672
    PYY_new = -254.9480896
    PZZ_new = 128.575531

Vir_new = (VirXX_new + VirYY_new + VirZZ_new)/3.

P_new = (PXX_new + PYY_new + PZZ_new)/3.
        
Virdc_new = (KE/3-Pdc_new/2*Vbox/nm3tom3/kJm3tobar*NA)/3.
       
for ibasis in range(2):
    eps_rerun = eps_basis[ibasis]
    sig_rerun = sig_basis[ibasis]
    print('Rerun with epsilon = '+str(eps_rerun)+' and sigma = '+str(sig_rerun))       
    Clam[ibasis] = eps_rerun * sig_rerun ** lam_basis
    C6[ibasis] = eps_rerun * sig_rerun ** 6.
    #U_basis[ibasis] = 10. * Clam[ibasis] + 5. * C6[ibasis] # This will eventually be replaced by a simulation
    Cmatrix[ibasis,0] = Clam[ibasis]
    Cmatrix[ibasis,1] = C6[ibasis]
    P_calc = 2./Vbox*(KE/3.-Vir_basis[ibasis])*nm3tom3*kJm3tobar/NA
    assert np.abs(P_calc - P_basis[ibasis]) < 1e-3, 'Conversion to pressure has error'

rarray = np.linalg.solve(Cmatrix,U_basis) #First entry of rarray is the sum of r^-lambda and second entry is sum of r^-6
print(rarray)

#Attempted to just multiply by the r du/dr terms
#rdrarray = rarray.copy()
#rdrarray[0] *= lam_basis
#rdrarray[1] *= 6.
rdrarray = np.linalg.solve(Cmatrix,Vir_basis-Virdc_basis) #First entry of rdrarray is the sum of r*d(r^-lambda)/dr and second entry is sum of r*d(r^-6)/dr
print(rdrarray)

rdrXXarray = np.linalg.solve(Cmatrix,VirXX_basis-Virdc_basis) #First entry of rdrarray is the sum of r*d(r^-lambda)/dr and second entry is sum of r*d(r^-6)/dr
rdrYYarray = np.linalg.solve(Cmatrix,VirYY_basis-Virdc_basis) #First entry of rdrarray is the sum of r*d(r^-lambda)/dr and second entry is sum of r*d(r^-6)/dr
rdrZZarray = np.linalg.solve(Cmatrix,VirZZ_basis-Virdc_basis) #First entry of rdrarray is the sum of r*d(r^-lambda)/dr and second entry is sum of r*d(r^-6)/dr

Ulam = np.linalg.multi_dot([Clam,rarray[0]])
U6 = np.linalg.multi_dot([C6,rarray[1]])
VirXXlam = np.linalg.multi_dot([Clam,rdrXXarray[0]])
VirXX6 = np.linalg.multi_dot([C6,rdrXXarray[1]])

for ibasis in range(2):
    assert np.abs(Ulam[ibasis]+U6[ibasis] - U_basis[ibasis]) < 1e-6, 'Energies do not add up'
    assert np.abs(VirXXlam[ibasis]+VirXX6[ibasis]+Virdc_basis[ibasis] - VirXX_basis[ibasis]) < 1e-6, 'Pressures do not add up'

# Verified that N_basis prefactor does not matter
Clam_new = eps_new * sig_new ** lam_basis #I intentionally ommitted the prefactor (4 for LJ) and just lumped it into the rarray
C6_new = eps_new * sig_new ** 6.

Cmatrix_new = np.array([Clam_new,C6_new])

U_new_hat = np.linalg.multi_dot([Cmatrix_new,rarray])
print('Predicted internal energy: '+str(U_new_hat))
print('Actual internal energy is '+str(U_new))

Vir_new_hat = np.linalg.multi_dot([Cmatrix_new,rdrarray])+Virdc_new
print('Predicted virial: '+str(Vir_new_hat))
print('Actual virial is '+str(Vir_new))

P_new_hat = 2./Vbox*(KE/3.-Vir_new_hat)*nm3tom3*kJm3tobar/NA
print('Predicted pressure: '+str(P_new_hat))
print('Actual pressure is '+str(P_new))

VirXX_new_hat = np.linalg.multi_dot([Cmatrix_new,rdrXXarray])+Virdc_new
print('Predicted virial-XX: '+str(VirXX_new_hat))
print('Actual virial-XX is '+str(VirXX_new))

VirYY_new_hat = np.linalg.multi_dot([Cmatrix_new,rdrYYarray])+Virdc_new
print('Predicted virial-YY: '+str(VirYY_new_hat))
print('Actual virial-YY is '+str(VirYY_new))

VirZZ_new_hat = np.linalg.multi_dot([Cmatrix_new,rdrZZarray])+Virdc_new
print('Predicted virial-ZZ: '+str(VirZZ_new_hat))
print('Actual virial-ZZ is '+str(VirZZ_new))

Vir_new_hat_alt = (VirXX_new_hat + VirYY_new_hat + VirZZ_new_hat)/3.
                  
P_new_hat_alt = 2./Vbox*(KE/3.-Vir_new_hat_alt)*nm3tom3*kJm3tobar/NA
print('Predicted pressure alternative: '+str(P_new_hat_alt))
print('Actual pressure is '+str(P_new))