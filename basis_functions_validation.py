# -*- coding: utf-8 -*-
"""
This code is designed to assess the performance of the basis set approach
with large numbers of configurations.
This code has not been optimized in order to be clear.

"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

nm3tom3 = 1e27
kJm3tobar = 1./100.
NA = 6.022140857e23
  
# Constants from state point
Lbox = 3.2168 #[nm]
Vbox = Lbox**3. #[nm3]

xvg_rr1 = np.loadtxt('energy_press_ref0rr1.txt')
xvg_rr2 = np.loadtxt('energy_press_ref0rr2.txt')
xvg_rr3 = np.loadtxt('energy_press_ref0rr3.txt')
xvg_rr4 = np.loadtxt('energy_press_ref0rr4.txt')

n_snaps = len(xvg_rr1)

eps_basis = np.zeros(2)
sig_basis = np.zeros(2)
Clam = np.zeros(2)
C6 = np.zeros(2)
U_basis = np.zeros([2,n_snaps])
Vir_basis = np.zeros([2,n_snaps])
VirXX_basis = np.zeros([2,n_snaps])
VirYY_basis = np.zeros([2,n_snaps])
VirZZ_basis = np.zeros([2,n_snaps])
Pdc_basis = np.zeros([2,n_snaps])
P_basis = np.zeros([2,n_snaps])
PXX_basis = np.zeros([2,n_snaps])
PYY_basis = np.zeros([2,n_snaps])
PZZ_basis = np.zeros([2,n_snaps])
Cmatrix = np.zeros([2,2])

#Taken from LJtoMie_Ufirst_lam13 ref0 Isochore rho0 T0, reruns 1,2

eps_basis[0] = 104.466306711
eps_basis[1] = 116.387441424   
sig_basis[0] = 0.375138117621
sig_basis[1] = 0.375768262189
         
lam_basis = 13.0
N_basis = (lam_basis/(lam_basis-6.))*(lam_basis/6.)**(6./(lam_basis-6.))

# Constrained with LINCS
# rr1 and rr2 are the two basis functions
U_basis[0] =  xvg_rr1[:,0]
U_basis[1] =  xvg_rr2[:,0]
       
#Virial XX,YY,ZZ
VirXX_basis[0] = xvg_rr1[:,5]
VirXX_basis[1] = xvg_rr2[:,5]
VirYY_basis[0] = xvg_rr1[:,6]
VirYY_basis[1] = xvg_rr2[:,6]
VirZZ_basis[0] = xvg_rr1[:,7]
VirZZ_basis[1] = xvg_rr2[:,7]
   
#Pressure XX,YY,ZZ
PXX_basis[0] = xvg_rr1[:,8]
PXX_basis[1] = xvg_rr2[:,8]    
PYY_basis[0] = xvg_rr1[:,9]  
PYY_basis[1] = xvg_rr2[:,9]
PZZ_basis[0] = xvg_rr1[:,10]
PZZ_basis[1] = xvg_rr2[:,10] 
         
P_basis[0] = xvg_rr1[:,4]
P_basis[1] = xvg_rr2[:,4]

#KE is constant with force field
KE = xvg_rr1[:,2]

      
#Virial and pressure average (with respect to XX,YY,ZZ-not time)
for ibasis in range(2):
    Vir_basis[ibasis] = (VirXX_basis[ibasis]+VirYY_basis[ibasis]+VirZZ_basis[ibasis])/3.
    assert np.all(np.abs(P_basis[ibasis] - (PXX_basis[ibasis]+PYY_basis[ibasis]+PZZ_basis[ibasis])/3.) < 1e-3), 'Error in pressure inputs'
       
#Dispersive contributions
Pdc_basis[0] = xvg_rr1[0,3]
Pdc_basis[1] = xvg_rr2[0,3]  
     
Vir_basis_alt = KE/3.-P_basis/2.*Vbox/nm3tom3/kJm3tobar*NA
Virdc_basis = (KE/3.-Pdc_basis/2.*Vbox/nm3tom3/kJm3tobar*NA)/3. #Dividing by 3 is an important step
       
# Values for a different force field, not used in the basis function development

rr_all = [3,4]

dev_U = np.zeros([len(rr_all),n_snaps])
dev_Vir = np.zeros([len(rr_all),n_snaps])
dev_P = np.zeros([len(rr_all),n_snaps])
mean_dev_U = np.zeros(len(rr_all))
mean_dev_Vir = np.zeros(len(rr_all))
mean_dev_P = np.zeros(len(rr_all))
per_bias_U = np.zeros(len(rr_all))
per_bias_Vir = np.zeros(len(rr_all))
per_bias_P = np.zeros(len(rr_all))

for rr_i, rr_new in enumerate(rr_all):

    if rr_new == 3:
        
        eps_new = 101.250841825
        sig_new = 0.37511601053
        
        # Constrained LINCS
        
        U_new = xvg_rr3[:,0]
        VirXX_new = xvg_rr3[:,5]
        VirYY_new = xvg_rr3[:,6]
        VirZZ_new = xvg_rr3[:,7]
        Pdc_new = xvg_rr3[:,3]
        PXX_new = xvg_rr3[:,8]
        PYY_new = xvg_rr3[:,9]
        PZZ_new = xvg_rr3[:,10]
        P_new = xvg_rr3[:,4]
        
    elif rr_new == 4:
        
        eps_new = 99.51946303	
        sig_new = 0.376196595
        
        # Constrained LINCS
        
        U_new = xvg_rr4[:,0]
        VirXX_new = xvg_rr4[:,5]
        VirYY_new = xvg_rr4[:,6]
        VirZZ_new = xvg_rr4[:,7]
        Pdc_new = xvg_rr4[:,3]
        PXX_new = xvg_rr4[:,8]
        PYY_new = xvg_rr4[:,9]
        PZZ_new = xvg_rr4[:,10]
        P_new = xvg_rr4[:,4]
        
    else:
        
        print('Not a valid rr')
           
    Vir_new = (VirXX_new + VirYY_new + VirZZ_new)/3. #Dividing by 3 is an important step
    
    assert np.all(np.abs(P_new - (PXX_new + PYY_new + PZZ_new)/3.) < 1e-3), 'Error in pressure input for new system'
            
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
        assert np.all(np.abs(P_calc - P_basis[ibasis]) < 1e-3), 'Conversion to pressure has error'
    
    rarray = np.linalg.solve(Cmatrix,U_basis) #First entry of rarray is the sum of r^-lambda and second entry is sum of r^-6
    print(rarray)
    
    rdrarray = np.linalg.solve(Cmatrix,Vir_basis-Virdc_basis) #First entry of rdrarray is the sum of r*d(r^-lambda)/dr and second entry is sum of r*d(r^-6)/dr
    print(rdrarray)
    
    rdrXXarray = np.linalg.solve(Cmatrix,VirXX_basis-Virdc_basis) #First entry of rdrarray is the sum of r*d(r^-lambda)/dr and second entry is sum of r*d(r^-6)/dr
    rdrYYarray = np.linalg.solve(Cmatrix,VirYY_basis-Virdc_basis) #First entry of rdrarray is the sum of r*d(r^-lambda)/dr and second entry is sum of r*d(r^-6)/dr
    rdrZZarray = np.linalg.solve(Cmatrix,VirZZ_basis-Virdc_basis) #First entry of rdrarray is the sum of r*d(r^-lambda)/dr and second entry is sum of r*d(r^-6)/dr
    
    for ibasis in range(2):
        Ulam = np.linalg.multi_dot([Clam[ibasis],rarray[0,:]])
        U6 = np.linalg.multi_dot([C6[ibasis],rarray[1,:]])
        VirXXlam = np.linalg.multi_dot([Clam[ibasis],rdrXXarray[0,:]])
        VirXX6 = np.linalg.multi_dot([C6[ibasis],rdrXXarray[1,:]])
    
        assert np.all(np.abs(Ulam+U6 - U_basis[ibasis]) < 1e-6), 'Energies do not add up'
        assert np.all(np.abs(VirXXlam+VirXX6+Virdc_basis[ibasis] - VirXX_basis[ibasis]) < 1e-6), 'Pressures do not add up'
    
    # Verified that N_basis prefactor does not matter
    Clam_new = eps_new * sig_new ** lam_basis #I intentionally ommitted the prefactor (4 for LJ) and just lumped it into the rarray
    C6_new = eps_new * sig_new ** 6.
    
    Cmatrix_new = np.array([Clam_new,C6_new])
    
    U_new_hat = np.linalg.multi_dot([Cmatrix_new,rarray])
    #print('Predicted internal energy: '+str(U_new_hat))
    #print('Actual internal energy is '+str(U_new))
    
    Vir_new_hat = np.linalg.multi_dot([Cmatrix_new,rdrarray])+Virdc_new
    #print('Predicted virial: '+str(Vir_new_hat))
    #print('Actual virial is '+str(Vir_new))
    
    P_new_hat = 2./Vbox*(KE/3.-Vir_new_hat)*nm3tom3*kJm3tobar/NA
    #print('Predicted pressure: '+str(P_new_hat))
    #print('Actual pressure is '+str(P_new))
    
    VirXX_new_hat = np.linalg.multi_dot([Cmatrix_new,rdrXXarray])+Virdc_new
    #print('Predicted virial-XX: '+str(VirXX_new_hat))
    #print('Actual virial-XX is '+str(VirXX_new))
    
    VirYY_new_hat = np.linalg.multi_dot([Cmatrix_new,rdrYYarray])+Virdc_new
    #print('Predicted virial-YY: '+str(VirYY_new_hat))
    #print('Actual virial-YY is '+str(VirYY_new))
    
    VirZZ_new_hat = np.linalg.multi_dot([Cmatrix_new,rdrZZarray])+Virdc_new
    #print('Predicted virial-ZZ: '+str(VirZZ_new_hat))
    #print('Actual virial-ZZ is '+str(VirZZ_new))
    
    Vir_new_hat_alt = (VirXX_new_hat + VirYY_new_hat + VirZZ_new_hat)/3.
                      
    P_new_hat_alt = 2./Vbox*(KE/3.-Vir_new_hat_alt)*nm3tom3*kJm3tobar/NA
    #print('Predicted pressure alternative: '+str(P_new_hat_alt))
    #print('Actual pressure is '+str(P_new))
    
    assert np.all(np.abs(Vir_new_hat_alt - Vir_new_hat) < 1e-3), 'Two methods for virial are different'
    assert np.all(np.abs(P_new_hat_alt - P_new_hat) < 1e-3), 'Two methods for pressure are different'
    
    dev_U[rr_i] = U_new_hat - U_new
    dev_Vir[rr_i] = Vir_new_hat - Vir_new
    dev_P[rr_i] = P_new_hat - P_new
    
    mean_dev_U[rr_i] = np.mean(dev_U[rr_i])
    mean_dev_Vir[rr_i] = np.mean(dev_Vir[rr_i])
    mean_dev_P[rr_i] = np.mean(dev_P[rr_i])
    
    per_bias_U[rr_i] = mean_dev_U[rr_i] / np.mean(U_new) * 100.
    per_bias_Vir[rr_i] = mean_dev_Vir[rr_i] / np.mean(Vir_new) * 100.
    per_bias_P[rr_i] = mean_dev_P[rr_i] / np.mean(P_new) * 100.
    
    U_parity = np.array([np.min(U_new),np.max(U_new)])
    Vir_parity = np.array([np.min(Vir_new),np.max(Vir_new)])
    P_parity = np.array([np.min(P_new),np.max(P_new)])
    
    plt.plot(U_new,U_new_hat,'ro')
    plt.plot(U_parity,U_parity,'k--')
    plt.xlabel('Simulated Energies (kJ/mol)')
    plt.ylabel('Basis Function Energies (kJ/mol)')
    plt.show()
    
    plt.plot(Vir_new,Vir_new_hat,'ro')
    #plt.plot(Vir_new,Vir_new_hat_alt,'bx')
    plt.plot(Vir_parity,Vir_parity,'k--')
    plt.xlabel('Simulated Virial (kJ/mol)')
    plt.ylabel('Basis Function Virial (kJ/mol)')
    plt.show()
    
    plt.plot(P_new,P_new_hat,'ro')
    #plt.plot(Vir_new,Vir_new_hat_alt,'bx')
    plt.plot(P_parity,P_parity,'k--')
    plt.xlabel('Simulated Pressure (bar)')
    plt.ylabel('Basis Function Pressure (bar)')
    plt.show()
    
    print('The average deviation in energy is '+str(mean_dev_U[rr_i])+' (kJ/mol), i.e. '+str(per_bias_U[rr_i])+'% bias')
    print('The average deviation in virial is '+str(mean_dev_Vir[rr_i])+' (kJ/mol), i.e. '+str(per_bias_Vir[rr_i])+'% bias')
    print('The average deviation in pressure is '+str(mean_dev_P[rr_i])+' (bar), i.e. '+str(per_bias_P[rr_i])+'% bias')
    
plt.plot(dev_U[0],'ro',label='rr3')
plt.plot(dev_U[1],'bx',label='rr4')
plt.xlabel('Snapshot')
plt.ylabel('Deviation in energy (kJ/mol)')
plt.legend()
plt.show()

plt.plot(dev_Vir[0],'ro',label='rr3')
plt.plot(dev_Vir[1],'bx',label='rr4')
plt.xlabel('Snapshot')
plt.ylabel('Deviation in virial (kJ/mol)')
plt.legend()
plt.show()

plt.plot(dev_P[0],'ro',label='rr3')
plt.plot(dev_P[1],'bx',label='rr4')
plt.xlabel('Snapshot')
plt.ylabel('Deviation in pressure (bar)')
plt.legend()
plt.show()
