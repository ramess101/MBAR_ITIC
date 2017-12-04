# -*- coding: utf-8 -*-
"""
Class for basis function generation

@author: ram9
"""

from __future__ import division
import numpy as np 
import os, sys, argparse, shutil
from pymbar import MBAR
from pymbar import timeseries
import CoolProp.CoolProp as CP
import subprocess
import time
from scipy.optimize import minimize, minimize_scalar, fsolve
import scipy.integrate as integrate
from create_tab import convert_eps_sig_C6_Clam

#Before running script run, "pip install pymbar, pip install CoolProp"

compound='ETHANE'
#compound='Ethane'
REFPROP_path='/home/ram9/REFPROP-cmake/build/' #Change this for a different system

CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH,REFPROP_path)

Mw = CP.PropsSI('M','REFPROP::'+compound) #[kg/mol]
RP_TC = CP.PropsSI('TCRIT','REFPROP::'+compound)
RP_Tmin =  CP.PropsSI('TMIN','REFPROP::'+compound)


# Physical constants
N_A = 6.02214086e23 #[/mol]
nm3_to_ml = 10**21
nm3_to_m3 = 10**27
bar_nm3_to_kJ_per_mole = 0.0602214086
R_g = 8.3144598 / 1000. #[kJ/mol/K]
kJm3tobar = 1./100.

# Simulation system specifications

rc = 1.4 #[nm]
N_sites = 2
N_inter = N_sites**2

#Read in the simulation specifications

ITIC = np.array(['Isotherm', 'Isochore'])
Temp_ITIC = {'Isochore':[],'Isotherm':[]}
rho_ITIC = {'Isochore':[],'Isotherm':[]}
Nmol = {'Isochore':[],'Isotherm':[]}
Temps = {'Isochore':[],'Isotherm':[]}
rhos_ITIC = {'Isochore':[],'Isotherm':[]}
rhos_mass_ITIC = {'Isochore':[],'Isotherm':[]}
nTemps = {'Isochore':[],'Isotherm':[]}
nrhos = {'Isochore':[],'Isotherm':[]}

Temp_sim = np.empty(0)
rho_sim = np.empty(0)
Nmol_sim = np.empty(0)

#Extract state points from ITIC files
# Move this outside of this loop so that we can just call it once, also may be easier for REFPROP
# Then again, with ITIC in the future Tsat will depend on force field

for run_type in ITIC:

    run_type_Settings = np.loadtxt(run_type+'Settings.txt',skiprows=1)

    Nmol[run_type] = run_type_Settings[:,0]
    Lbox = run_type_Settings[:,1] #[nm]
    Temp_ITIC[run_type] = run_type_Settings[:,2] #[K]
    Vol = Lbox**3 #[nm3]
    rho_ITIC[run_type] = Nmol[run_type] / Vol #[molecules/nm3]
    rhos_ITIC[run_type] = np.unique(rho_ITIC[run_type])
    rhos_mass_ITIC[run_type] = rhos_ITIC[run_type] * Mw / N_A * nm3_to_m3 #[kg/m3]
    nrhos[run_type] = len(rhos_ITIC[run_type])
    Temps[run_type] = np.unique(Temp_ITIC[run_type])
    nTemps[run_type] = len(Temps[run_type]) 
 
    Temp_sim = np.append(Temp_sim,Temp_ITIC[run_type])
    rho_sim = np.append(rho_sim,rho_ITIC[run_type])
    Nmol_sim = np.append(Nmol_sim,Nmol[run_type])

nTemps['Isochore']=2 #Need to figure out how to get this without hardcoding
    
rho_mass = rho_sim * Mw / N_A * nm3_to_m3 #[kg/m3]
#rho_mass = rho_sim * Mw / N_A * nm3_to_ml #[gm/ml]

nStates = len(Temp_sim)

# Create a list of all the file paths (without the reference directory, just the run_type, rho, Temp)

fpath_all = []

for run_type in ITIC: 

    for irho  in np.arange(0,nrhos[run_type]):

        for iTemp in np.arange(0,nTemps[run_type]):

            if run_type == 'Isochore':

                fpath_all.append(run_type+'/rho'+str(irho)+'/T'+str(iTemp)+'/NVT_eq/NVT_prod/')

            else:

                fpath_all.append(run_type+'/rho_'+str(irho)+'/NVT_eq/NVT_prod/')
                
assert nStates == len(fpath_all), 'Number of states does not match number of file paths'

class basis_function():
    def __init__(self,Temp_sim,rho_sim,iRef,iRefs,eps_low,eps_high,sig_low,sig_high,lam_low,lam_high,rerun_flag=True):
        self.Temp_sim = Temp_sim
        self.rho_sim = rho_sim
        self.iRef = iRef
        self.iRefs = iRefs
        self.nRefs = len(iRefs)
        
        self.eps_high = eps_high
        self.eps_low = eps_low
        self.sig_high = sig_high
        self.sig_low = sig_low
        self.lam_high = lam_high
        self.lam_low = lam_low
        self.rerun_flag = rerun_flag
        
        self.eps_sig_lam_refs = self.compile_refs()
        self.eps_sig_lam_ref = self.eps_sig_lam_refs[0]
        self.eps_ref = self.eps_sig_lam_ref[0]
        self.sig_ref = self.eps_sig_lam_ref[1]
        self.lam_ref = self.eps_sig_lam_ref[2]
        
        self.eps_sig_lam_basis, self.eps_basis, self.sig_basis, self.lam_basis = self.create_eps_sig_lam_basis()
        self.Cmatrix, self.lam_index = self.create_Cmatrix()
        self.rerun_basis_functions()        
        self.generate_basis_functions()
        self.calc_refs_basis()
        self.validate_refs()
        
    def compile_refs(self):
        
        iRef,nRefs,iRefs = self.iRef,self.nRefs,self.iRefs
     
        eps_sig_lam_refs = np.zeros([nRefs,3])
        
        for iiiRef, iiRef in enumerate(iRefs):
            
            eps_sig_lam_refs[iiiRef,:] = np.loadtxt('../ref'+str(iiRef)+'/eps_sig_lam_ref')
            
        return eps_sig_lam_refs
        
    def create_eps_sig_lam_basis(self):
        
        eps_ref,sig_ref,lam_ref,eps_low,eps_high,sig_low,sig_high,lam_low,lam_high = self.eps_ref,self.sig_ref,self.lam_ref,self.eps_low,self.eps_high,self.sig_low,self.sig_high,self.lam_low,self.lam_high
    
    # I am going to keep it the way I had it where I just submit with real parameters 
    # And solve linear system of equations. 
    #    print(lam_low)
    #    print(lam_high)
        nBasis = len(range(int(lam_low),int(lam_high)+1))+1 #The 2 is necessary if only using LJ 12-6
        
        eps_basis = np.ones(nBasis)*eps_low
        sig_basis = np.ones(nBasis)*sig_low 
        lam_basis = np.ones(nBasis)*lam_ref # Should be 12 to start
        
        eps_basis[0] = eps_high
        sig_basis[0] = sig_high
#        eps_basis[1] = eps_ref
#        sig_basis[1] = sig_ref
        lam_basis[1:] = range(int(lam_low),int(lam_high)+1) 
        
    #    print(eps_basis)
    #    print(sig_basis)
    #    print(lam_basis)
    
        eps_sig_lam_basis = np.array([eps_basis,sig_basis,lam_basis]).transpose()
        
        return eps_sig_lam_basis, eps_basis, sig_basis, lam_basis

    def create_Cmatrix(self):
        '''
        This function creates the matrix of C6, C12, C13... where C6 is always the
        zeroth column and the rest is the range of lambda from lam_low to lam_high.
        These values were predefined in rerun_basis_functions
        '''
        
        iRef,eps_basis,sig_basis,lam_basis = self.iRef,self.eps_basis.copy(),self.sig_basis.copy(),self.lam_basis.copy()
        
        Cmatrix = np.zeros([len(lam_basis),len(lam_basis)])
        
        C6_basis, Clam_basis = convert_eps_sig_C6_Clam(eps_basis,sig_basis,lam_basis,print_Cit=False)
        
        lam_index = lam_basis.copy()
        lam_index[0] = 6
    
        Cmatrix[:,0] = C6_basis
               
        for ilam, lam in enumerate(lam_index):
            for iBasis, lam_rerun in enumerate(lam_basis):
                if lam == lam_rerun:
                    Cmatrix[iBasis,ilam] = Clam_basis[iBasis]
                    
#        fpath = "../ref"+str(iRef)+"/"
#        
#        f = open(fpath+'Cmatrix','w')
#        g = open(fpath+'lam_index','w')
#        
#        for ilam, lam in enumerate(lam_index):
#            for jlam in range(len(lam_basis)):
#                f.write(str(Cmatrix[ilam,jlam])+'\t')
#            f.write('\n')
#            g.write(str(lam)+'\t')              
#        
#        f.close()
#        g.close()       
                
        return Cmatrix, lam_index 
    
    def check_basis_functions_U(self,LJsr,sumr6lam,Cmatrix):
        LJhat = np.linalg.multi_dot([Cmatrix,sumr6lam])
        assert (np.abs(LJsr - LJhat) < 1e-3).all(), 'Basis functions for internal energy deviate by at most:'+str(np.max(np.abs(LJsr-LJhat)))
        
    def check_basis_functions_Vir(self,Vir_0,Vir_1,Vir_2,sumrdr6lam_vdw,sumrdr6lam_LINCS,Cmatrix):
        Vir_vdw_hat = np.linalg.multi_dot([Cmatrix,sumrdr6lam_vdw])
        Vir_LINCS_hat = np.linalg.multi_dot([Cmatrix,sumrdr6lam_LINCS])
        Vir_total_hat = Vir_vdw_hat + Vir_LINCS_hat + Vir_2
        
        assert (np.abs(Vir_1 - Vir_vdw_hat) < 1e-3).all(), 'Basis function for vdw virial deviates at most by:'+str(np.max(np.abs(Vir_1-Vir_vdw_hat)))
 
        assert (np.abs(Vir_0 - Vir_1 - Vir_2 - Vir_LINCS_hat) < 1e-3).all(), 'Basis function for LINCS virial deviates at most by:'+str(np.np.max(np.abs(Vir_0 - Vir_1 - Vir_2 -Vir_LINCS_hat)))
    
        assert (np.abs(Vir_0 - Vir_total_hat) < 1e-3).all(), 'Basis function for virial deviates at most by:'+str(np.np.max(np.abs(Vir_0 - Vir_total_hat)))
        
        return Vir_vdw_hat, Vir_LINCS_hat, Vir_total_hat
        
    def check_basis_functions_press(self,press,p_0,p_1,p_2,Vir_vdw_hat, Vir_LINCS_hat, Vir_total_hat,KE,iState):
        
        press_vdw_hat = self.convert_VirialtoP(KE,Vir_vdw_hat,iState)
        press_LINCS_hat = self.convert_VirialtoP(KE,Vir_LINCS_hat,iState)
        press_total_hat = self.convert_VirialtoP(KE,Vir_total_hat,iState)
        
        press_dev = np.abs(press-press_total_hat)
                
        assert (press_dev < 1e-3).all(), 'Basis functions for the total pressure deviate by at most:'+str(np.max(press_dev))
    
    def rerun_basis_functions(self):
        ''' 
    This function submits the rerun simulations that are necessary to generate the
    basis functions. It starts by performing a rerun simulation with the LJ model 
    at the highest epsilon and sigma. It then submits a rerun using LJ with the 
    lowest epsilon and sigma. Then, it submits a single rerun for all the different
    Mie values for lambda. 
        '''
    
        iRef,eps_basis,sig_basis,lam_basis,rerun_flag = self.iRef,self.eps_basis,self.sig_basis,self.lam_basis,self.rerun_flag
        
        iRerun = 0
        
        iRerun_basis = iRerun
                   
        for eps_rerun, sig_rerun, lam_rerun in zip(eps_basis, sig_basis, lam_basis):
            
            fpathRef = "../ref"+str(iRef)+"/"
            #print(fpathRef)
        
            f = open(fpathRef+'eps_it','w')
            f.write(str(eps_rerun))
            f.close()
        
            f = open(fpathRef+'sig_it','w')
            f.write(str(sig_rerun))
            f.close()
        
            f = open(fpathRef+'lam_it','w')
            f.write(str(lam_rerun))
            f.close()
            
            f = open(fpathRef+'iRerun','w')
            f.write(str(iRerun_basis))
            f.close()
            
            if rerun_flag:
            
                subprocess.call(fpathRef+"EthaneRerunITIC_basis")
    
            iRerun_basis += 1
            
            #print('iRerun is = '+str(iRerun)+', while iRerun_basis = '+str(iRerun_basis))
            
        f = open(fpathRef+'iBasis','w')
        f.write(str(iRerun_basis-1))
        f.close()
            
        self.iBasis = range(iRerun,iRerun_basis)
    
    def generate_basis_functions(self):
        
        iRef, iBasis, Cmatrix = self.iRef, self.iBasis,self.Cmatrix
        
        nSets = len(iBasis)
        
        g_start = 28 #Row where data starts in g_energy output
        g_t = 0 #Column for the snapshot time
        g_LJsr = 1 #Column where the 'Lennard-Jones' short-range interactions are located
        g_LJdc = 2 #Column where the 'Lennard-Jones' dispersion corrections are located
        g_en = 3 #Column where the potential energy is located
        g_KE = 4 #Column where KE is located
        g_p = 5 #Column where p is located
        
        iState = 0
        
        for run_type in ITIC: 
        
            for irho in np.arange(0,nrhos[run_type]):
        
                for iTemp in np.arange(0,nTemps[run_type]):
        
                    if run_type == 'Isochore':
        
                        fpath = run_type+'/rho'+str(irho)+'/T'+str(iTemp)+'/NVT_eq/NVT_prod/'
        
                    else:
        
                        fpath = run_type+'/rho_'+str(irho)+'/NVT_eq/NVT_prod/'
                    
                      
                    en_p = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%s.xvg' %(iRef,iRef),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
                        
                    nSnaps = len(en_p)
                    #print(N_k)
                    #print(nSnaps)
                    
                    if iState == 0:
                        KE_state = np.zeros([nStates,nSnaps])
                        Vir_novdw_state = np.zeros([nStates,nSnaps])
                        sumr6lam_state = np.zeros([nStates,nSets,nSnaps])
                        sumrdr6lam_vdw_state = np.zeros([nStates,nSets,nSnaps])
                        sumrdr6lam_LINCS_state = np.zeros([nStates,nSets,nSnaps])
                        U_novdw_state = np.zeros([nStates,nSnaps])
                        
                    t = np.zeros([nSets,nSnaps])
                    LJsr = np.zeros([nSets,nSnaps])
                    LJdc = np.zeros([nSets,nSnaps])
                    en = np.zeros([nSets,nSnaps])
                    p = np.zeros([nSets,nSnaps])
                    U_total = np.zeros([nSets,nSnaps])
                    LJ_total = np.zeros([nSets,nSnaps])
                    p_0 = np.zeros([nSets,nSnaps])
                    p_1 = np.zeros([nSets,nSnaps])
                    p_2 = np.zeros([nSets,nSnaps])
                    Vir_0 = np.zeros([nSets,nSnaps])
                    Vir_1 = np.zeros([nSets,nSnaps])
                    Vir_2 = np.zeros([nSets,nSnaps])
                    KE = np.zeros([nSets,nSnaps])
        
                    for iSet, enum in enumerate(iBasis):
                            
                        en_p = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%sbasis0.xvg' %(iRef,enum),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
                      
                        for frame in xrange(nSnaps):
                            t[iSet][frame] = float(en_p[frame].split()[g_t])
                            LJsr[iSet][frame] = float(en_p[frame].split()[g_LJsr])
                            LJdc[iSet][frame] = float(en_p[frame].split()[g_LJdc])
                            en[iSet][frame] = float(en_p[frame].split()[g_en])
                            p[iSet][frame] = float(en_p[frame].split()[g_p])
                            KE[iSet][frame] = float(en_p[frame].split()[g_KE])
                            #f.write(str(p[iSet][frame])+'\n')
    
                        U_total[iSet] = en[iSet] # For TraPPEfs we just used potential because dispersion was erroneous. I believe we still want potential even if there are intramolecular contributions. 
                        LJ_total[iSet] = LJsr[iSet] + LJdc[iSet] #In case we want just the LJ total (since that would be U_res as long as no LJ intra). We would still use U_total for MBAR reweighting but LJ_total would be the observable
                       
                    for iSet, enum in enumerate(iBasis):
                            
                        en_p_0 = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%sbasis0.xvg' %(iRef,enum),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
                        en_p_1 = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%sbasis1.xvg' %(iRef,enum),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
                        en_p_2 = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr0basis2.xvg' %(iRef),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
                        
                        for frame in xrange(nSnaps):
                            p_0[iSet][frame] = float(en_p_0[frame].split()[g_p])
                            p_1[iSet][frame] = float(en_p_1[frame].split()[g_p])
                            p_2[iSet][frame] = float(en_p_2[frame].split()[g_p])      
        
                        Vir_0[iSet] = self.convert_PtoVirial(KE[iSet],p_0[iSet],iState)
                        Vir_1[iSet] = self.convert_PtoVirial(KE[iSet],p_1[iSet],iState)
                        Vir_2[iSet] = self.convert_PtoVirial(KE[iSet],p_2[iSet],iState)
                                 
                    sumr6lam = np.zeros([nSets,nSnaps])
                    sumrdr6lam_vdw = np.zeros([nSets,nSnaps])
                    sumrdr6lam_LINCS = np.zeros([nSets,nSnaps])
                    
#                    f0 = open('../ref'+str(iRef)+'/'+fpath+'basis_functions_U','w')
#                    f1 = open('../ref'+str(iRef)+'/'+fpath+'basis_functions_vir_vdw','w')
#                    f2 = open('../ref'+str(iRef)+'/'+fpath+'basis_functions_vir_LINCS','w')
#                    f3 = open('../ref'+str(iRef)+'/'+fpath+'kinetic_energy','w')
#                    f4 = open('../ref'+str(iRef)+'/'+fpath+'virial_novdw','w')
    
                    for frame in xrange(nSnaps):
                        U_basis_vdw = LJsr[:,frame]
                        Vir_basis_vdw = Vir_1[:,frame]
                        Vir_basis_LINCS = Vir_0[:,frame] - Vir_1[:,frame] - Vir_2[:,frame]
                        sumr6lam[:,frame] = np.linalg.solve(Cmatrix,U_basis_vdw)
                        sumrdr6lam_vdw[:,frame] = np.linalg.solve(Cmatrix,Vir_basis_vdw)
                        sumrdr6lam_LINCS[:,frame] = np.linalg.solve(Cmatrix,Vir_basis_LINCS)
                        
                        assert sumr6lam[0,frame] < 0, 'The attractive contribution has the wrong sign'
                    
#                        for iSet in range(nSets):
#                            if iSet < nSets-1:
#                                f0.write(str(sumr6lam[iSet,frame])+'\t')
#                                f1.write(str(sumrdr6lam_vdw[iSet,frame])+'\t')
#                                f2.write(str(sumrdr6lam_LINCS[iSet,frame])+'\t')
#                            else:
#                                f0.write(str(sumr6lam[iSet,frame])+'\n')
#                                f1.write(str(sumrdr6lam_vdw[iSet,frame])+'\n')
#                                f2.write(str(sumrdr6lam_LINCS[iSet,frame])+'\n')
#                                
#                        f3.write(str(KE[0,frame])+'\n')
#                        f4.write(str(Vir_2[0,frame])+'\n')
#                        
#                    f0.close()
#                    f1.close()
#                    f2.close()
#                    f3.close()
#                    f4.close()
                    
                    KE_state[iState,:] = KE[0,:]
                    Vir_novdw_state[iState,:] = Vir_2[0,:]
                    sumr6lam_state[iState,:,:] = sumr6lam
                    sumrdr6lam_vdw_state[iState,:,:] = sumrdr6lam_vdw
                    sumrdr6lam_LINCS_state[iState,:,:] = sumrdr6lam_LINCS
                    U_novdw_state[iState,:] = U_total[0,:] - LJ_total[0,:] #There are several ways to track U_novdw 
                    
                    self.check_basis_functions_U(LJsr,sumr6lam,Cmatrix)
                    Vir_vdw_hat, Vir_LINCS_hat, Vir_total_hat = self.check_basis_functions_Vir(Vir_0,Vir_1,Vir_2,sumrdr6lam_vdw,sumrdr6lam_LINCS,Cmatrix)
                    pcheck = self.convert_VirialtoP(KE,Vir_0,iState)
                    assert (np.abs(p-pcheck)< 1e-3).all(), 'Conversion of virial to P has deviations of at most:'+str(np.max(np.abs(p-pcheck)))
                    self.check_basis_functions_press(p,p_0,p_1,p_2,Vir_vdw_hat, Vir_LINCS_hat, Vir_total_hat,KE,iState)
                        
                    #self.sumr6lam[iState], self.sumrdr6lam_vdw[iState], self.sumrdr6lam_LINCS[iState] = sumr6lam, sumrdr6lam_vdw, sumrdr6lam_LINCS
                        
                    iState += 1   
                    
        self.KE_state, self.Vir_novdw_state, self.sumr6lam_state, self.sumrdr6lam_vdw_state, self.sumrdr6lam_LINCS_state, self.U_novdw_state = KE_state, Vir_novdw_state, sumr6lam_state, sumrdr6lam_vdw_state, sumrdr6lam_LINCS_state, U_novdw_state
                    
    def convert_PtoVirial(self,KE,press,iState):
        """ Calculate the virial
        KE: kinetic energy [kJ/mol]
        press: pressure [bar]
        iState: ITIC state point 
        
        Vir: virial [kJ/mol]
        """
        
        rho = self.rho_sim[iState] #[molecules/nm3]
        Vol = 1./rho/nm3_to_m3 #[m3/molecules]
        
        Vir = KE/3. - press/2.*Vol/kJm3tobar*N_A
        
        return Vir
    
    def convert_VirialtoP(self,KE,Vir,iState):
        """ Calculate the virial
        KE: kinetic energy [kJ/mol]
        Vir: virial [kJ/mol]
        iState: ITIC state point 
        
        press: pressure [bar]
        """
        
        rho = self.rho_sim[iState] #[molecules/m3]
        Vol = 1./rho/nm3_to_m3 #[m3/molecules]
                
        press = 2./Vol*(KE/3. - Vir)*kJm3tobar/N_A
        
        return press
                    
    def create_Carray(self,eps_sig_lam):
        '''
        This function creates a single column array of C6, C12, C13... where C6 is always the
        zeroth column and the rest are 0 except for the column that pertains to lambda.
        '''
        
        iRef, lam_index = self.iRef, self.lam_index
        
        eps = eps_sig_lam[0]
        sig = eps_sig_lam[1]
        lam = eps_sig_lam[2]
        
        Carray = np.zeros([len(lam_index)])
        
        C6, Clam = convert_eps_sig_C6_Clam(eps,sig,lam,print_Cit=False)
        
        Carray[0] = C6
               
        for ilam, lam_basis in enumerate(lam_index):
            if lam == lam_basis:
                Carray[ilam] = Clam
                    
#        fpath = "../ref"+str(iRef)+"/"
#        
#        f = open(fpath+'Carrayit','a')
#        
#        for ilam, Ci in enumerate(Carray):
#            f.write(str(Ci)+'\t')
#        
#        f.write('\n')              
#        f.close()
                
        return Carray
    
    def LJ_tail_corr(self,C6,rho,Nmol):
        '''
        Calculate the LJ tail correction to U using the Gromacs approach (i.e. only C6 is used)
        '''
        U_Corr = -2./3. * np.pi * C6 * rc**(-3.)
        U_Corr *= Nmol * rho * N_inter
        return U_Corr
                    
    def UP_basis_functions(self,iState,eps_sig_lam):
        '''
        iState: state point (integer)
        eps_sig_lam: array of epsilon [K], sigma [nm], and lambda
        
        LJ_SR: short-range Lennard-Jones energy [kJ/mol]
        LJ_dc: dispersive correction LJ [kJ/mol]
        press: pressure [bar]
        '''
        
        iRef = self.iRef
        rho_state = rho_sim[iState]
        Nstate = Nmol_sim[iState]        
        
        #sumr6lam = np.loadtxt('../ref'+str(iRef)+'/'+fpath+'basis_functions_U')
        #sumrdr6lam_vdw = np.loadtxt('../ref'+str(iRef)+'/'+fpath+'basis_functions_vir_LINCS')
        #sumrdr6lam_LINCS = np.loadtxt('../ref'+str(iRef)+'/'+fpath+'basis_functions_vir_vdw')
#        KE = np.loadtxt('../ref'+str(iRef)+'/'+fpath+'kinetic_energy')
#        Vir_novdw = np.loadtxt('../ref'+str(iRef)+'/'+fpath+'virial_novdw')
        sumr6lam = self.sumr6lam_state[iState,:,:].transpose()
        sumrdr6lam_vdw = self.sumrdr6lam_vdw_state[iState,:,:].transpose()
        sumrdr6lam_LINCS = self.sumrdr6lam_LINCS_state[iState,:,:].transpose()
        KE = self.KE_state[iState,:]
        Vir_novdw = self.Vir_novdw_state[iState,:]
        
        Carray = self.create_Carray(eps_sig_lam)
        C6 = Carray[0]
        LJ_SR = np.linalg.multi_dot([sumr6lam,Carray])
        LJ_dc = np.ones(len(LJ_SR))*self.LJ_tail_corr(C6,rho_state,Nstate) # Use Carray[0] to convert C6 into LJ_dc
        
        Vir_vdw = np.linalg.multi_dot([sumrdr6lam_vdw,Carray])
        Vir_LINCS = np.linalg.multi_dot([sumrdr6lam_LINCS,Carray])
        Vir_total = Vir_vdw + Vir_LINCS + Vir_novdw
        
        press = self.convert_VirialtoP(KE,Vir_total,iState)
        #print('The initial LJ energy is '+str(LJ_SR[0]))
        return LJ_SR, LJ_dc, press
    
    def UP_basis_states(self,eps_sig_lam):
        
        for iState in range(nStates):
            LJsr_basis, LJdc_basis, press_basis = self.UP_basis_functions(iState,eps_sig_lam)
            
            if iState == 0:
                nSnaps = len(LJsr_basis)
                LJ_total_basis_rr_state = np.zeros([nStates,nSnaps])
                press_basis_rr_state = np.zeros([nStates,nSnaps])
                U_total_basis_rr_state = np.zeros([nStates,nSnaps])
                
            LJ_total_basis_rr_state[iState,:] = LJsr_basis + LJdc_basis
            press_basis_rr_state[iState,:] = press_basis
            U_total_basis_rr_state[iState,:] = LJ_total_basis_rr_state[iState,:] + self.U_novdw_state[iState,:]
                             
#        self.LJ_total_basis_rr_state, self.press_basis_rr_state = LJ_total_basis_rr_state, press_basis_rr_state
        return LJ_total_basis_rr_state, U_total_basis_rr_state, press_basis_rr_state

    def calc_refs_basis(self):
        eps_sig_lam_refs, nRefs = self.eps_sig_lam_refs, self.nRefs
        
        #print(eps_sig_lam_refs)
        for iiRef, eps_sig_lam_ref in enumerate(eps_sig_lam_refs):
            
            #eps_sig_lam_ref = eps_sig_lam_refs[:,iiRef]
            
            #print(eps_sig_lam_ref)
            
            if iiRef == 0:
                
                LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state = self.UP_basis_states(eps_sig_lam_ref)
                nSnaps = LJ_total_basis_ref_state.shape[1]
                assert nStates == LJ_total_basis_ref_state.shape[0], "Number of states does not match dimension"
                LJ_total_basis_refs_state = np.zeros([nRefs,nStates,nSnaps])
                U_total_basis_refs_state = np.zeros([nRefs,nStates,nSnaps])
                press_basis_refs_state = np.zeros([nRefs,nStates,nSnaps])
            
            else:
            
                LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state = self.UP_basis_states(eps_sig_lam_ref)
            
            LJ_total_basis_refs_state[iiRef], U_total_basis_refs_state[iiRef], press_basis_refs_state[iiRef] = LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state
        
        self.LJ_total_basis_refs_state, self.U_total_basis_refs_state, self.press_basis_refs_state =  LJ_total_basis_refs_state, U_total_basis_refs_state, press_basis_refs_state                        
    
    def validate_ref(self):
        iRef, iRefs = self.iRef, self.iRefs

        for iiiRef, iiRef in enumerate(iRefs):

            if iRef == iiRef:

                LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state = self.LJ_total_basis_refs_state[iiiRef], self.U_total_basis_refs_state[iiiRef], self.press_basis_refs_state[iiiRef]
        
        g_start = 28 #Row where data starts in g_energy output
        g_en = 3 #Column where the potential energy is located
        g_p = 5 #Column where p is located
        
        APD_U = np.zeros(nStates)
        APD_LJ = np.zeros(nStates)
        APD_P = np.zeros(nStates)

#        self.UP_basis_states(eps_sig_lam_ref)
        
#        LJ_total_basis_ref_state = self.LJ_total_basis_rr_state 
#        press_basis_ref_state = self.press_basis_rr_state
        
        for iState in range(nStates):
            
            LJ_total_basis_ref = LJ_total_basis_ref_state[iState]
            U_total_basis_ref = U_total_basis_ref_state[iState]
            press_basis_ref = press_basis_ref_state[iState]
            
            fpath = fpath_all[iState]
            
            en_p = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%s.xvg' %(iRef,iRef),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k

            nSnaps = len(en_p)

            press_ref = np.zeros(nSnaps)
            LJ_total_ref = np.zeros(nSnaps)
            U_total_ref = np.zeros(nSnaps)
            
            for frame in range(nSnaps):
            
                LJ_total_ref[frame] = float(en_p[frame].split()[g_en])
                U_total_ref[frame] = float(en_p[frame].split()[g_en])
                press_ref[frame] = float(en_p[frame].split()[g_p])
            
            LJ_dev = (LJ_total_basis_ref - LJ_total_ref)/LJ_total_ref*100.
            U_dev = (U_total_basis_ref - U_total_ref)/U_total_ref*100.
            press_dev = (press_basis_ref - press_ref)/np.mean(press_ref)*100.
            
#            for LJ, press in zip(LJ_dev,press_dev):
#                print(LJ,press)
            APD_LJ[iState] = np.mean(LJ_dev)
            APD_U[iState] = np.mean(U_dev)
            APD_P[iState] = np.mean(press_dev)
#            print(np.mean(LJ_dev))
#            print(np.mean(press_dev))
 
        print('Average percent deviation in non-bonded energy from basis functions compared to reference simulations: '+str(np.mean(APD_LJ)))           
        print('Average percent deviation in internal energy from basis functions compared to reference simulations: '+str(np.mean(APD_U)))
        print('Average percent deviation in pressure from basis functions compared to reference simulations: '+str(np.mean(APD_P)))

    def validate_refs(self):
        
        iRef,iRefs = self.iRef,self.iRefs
        
        for iiiRef, iiRef in enumerate(iRefs):
    
            LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state = self.LJ_total_basis_refs_state[iiiRef], self.U_total_basis_refs_state[iiiRef], self.press_basis_refs_state[iiiRef]
            
            g_start = 28 #Row where data starts in g_energy output
            g_en = 3 #Column where the potential energy is located
            g_p = 5 #Column where p is located
            
            APD_U = np.zeros(nStates)
            APD_LJ = np.zeros(nStates)
            APD_P = np.zeros(nStates)
    
    #        self.UP_basis_states(eps_sig_lam_ref)
            
    #        LJ_total_basis_ref_state = self.LJ_total_basis_rr_state 
    #        press_basis_ref_state = self.press_basis_rr_state
            
            for iState in range(nStates):
                
                LJ_total_basis_ref = LJ_total_basis_ref_state[iState]
                U_total_basis_ref = U_total_basis_ref_state[iState]
                press_basis_ref = press_basis_ref_state[iState]
                
                fpath = fpath_all[iState]
                
                en_p = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%s.xvg' %(iRef,iiRef),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
    
                nSnaps = len(en_p)
    
                press_ref = np.zeros(nSnaps)
                LJ_total_ref = np.zeros(nSnaps)
                U_total_ref = np.zeros(nSnaps)
                
                for frame in range(nSnaps):
                
                    LJ_total_ref[frame] = float(en_p[frame].split()[g_en])
                    U_total_ref[frame] = float(en_p[frame].split()[g_en])
                    press_ref[frame] = float(en_p[frame].split()[g_p])
                
                LJ_dev = (LJ_total_basis_ref - LJ_total_ref)/LJ_total_ref*100.
                U_dev = (U_total_basis_ref - U_total_ref)/U_total_ref*100.
                press_dev = (press_basis_ref - press_ref)/np.mean(press_ref)*100.
                
    #            for LJ, press in zip(LJ_dev,press_dev):
    #                print(LJ,press)
                APD_LJ[iState] = np.mean(LJ_dev)
                APD_U[iState] = np.mean(U_dev)
                APD_P[iState] = np.mean(press_dev)
    #            print(np.mean(LJ_dev))
    #            print(np.mean(press_dev))
            #print('iRef= '+str(iRef))
            #print('iiRef= '+str(iiRef))
            #print('Average percent deviation in non-bonded energy from basis functions compared to reference simulations: '+str(np.mean(APD_LJ)))           
            #print('Average percent deviation in internal energy from basis functions compared to reference simulations: '+str(np.mean(APD_U)))
            #print('Average percent deviation in pressure from basis functions compared to reference simulations: '+str(np.mean(APD_P)))
            assert np.abs(np.mean(APD_LJ)) < 1e-3, 'Basis function non-bonded energy error too large: '+str(np.mean(APD_LJ))+' for: iRef= '+str(iRef)+' iiRef= '+str(iiRef)
            assert np.abs(np.mean(APD_U)) < 1e-3, 'Basis function internal energy error too large: '+str(np.mean(APD_U))+' for: iRef= '+str(iRef)+' iiRef= '+str(iiRef)
            assert np.abs(np.mean(APD_P)) < 1e-1, 'Basis function pressure error too large: '+str(np.mean(APD_P))+' for: iRef= '+str(iRef)+' iiRef= '+str(iiRef)

def UP_basis_mult_refs(basis):
    
    nRefs = len(basis) #Rather than providing nRefs as a variable we will just determine it from the size of basis
    
    for iiRef in range(nRefs): 
            
        LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state = basis[iiRef].LJ_total_basis_refs_state, basis[iiRef].U_total_basis_refs_state, basis[iiRef].press_basis_refs_state
        
        if iiRef == 0:
            
            #print(LJ_total_basis_ref_state.shape)
            nSnaps = LJ_total_basis_ref_state.shape[2]
        
            LJ_total_basis_refs = np.zeros([nRefs,nRefs,nStates,nSnaps])
            U_total_basis_refs = np.zeros([nRefs,nRefs,nStates,nSnaps])
            press_basis_refs = np.zeros([nRefs,nRefs,nStates,nSnaps])                                                                                                                                                                           
#            for jRef in range(args.nRefs):
                       
        LJ_total_basis_refs[iiRef] = LJ_total_basis_ref_state
        U_total_basis_refs[iiRef] = U_total_basis_ref_state
        press_basis_refs[iiRef] = press_basis_ref_state
                        
     # Validated that these values agreed with what is expected                   
#        print(LJ_total_basis_refs[0,0,0,:])
#        print(LJ_total_basis_refs[1,0,0,:])
#        print(LJ_total_basis_refs[1,1,0,:])
#        print(LJ_total_basis_refs.shape)

    return LJ_total_basis_refs, U_total_basis_refs, press_basis_refs

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-nRefs","--nRefs",type=int,help="set the integer value for the number of references")
    parser.add_argument("-iRef","--iRef",type=int,nargs='+',help="set the integer value for the reference")
    args = parser.parse_args()
    
    basis = []
    
#    for i in range(2):
#        
#        basis.append(basis_function(Temp_sim,rho_sim,args.iRef,88.,108.,0.365,0.385,12.,12.))
#        basis[i].validate_ref()
#   
    if args.nRefs:

        for iiRef in range(args.nRefs):
            basis.append(basis_function(Temp_sim,rho_sim,iiRef,args.nRefs,88.,108.,0.365,0.385,12.,14.,False))
            basis[iiRef].validate_refs()
            
        LJ_total_basis_refs, U_total_basis_refs, press_basis_refs = UP_basis_mult_refs(basis,args.nRefs)
            
    if args.iRef:
        
        for iiRef in args.iRef:
        
            basis.append(basis_function(Temp_sim,rho_sim,iiRef,1,88.,108.,0.365,0.385,12.,12.,True)) 
#            basis.append(basis_function(Temp_sim,rho_sim,iiRef,88.,108.,0.365,0.385,12.,18.)) 
            basis[0].validate_ref()
#            print(np.mean(basis[0].U_novdw_state))
#            print(np.max(np.abs(basis[0].U_novdw_state)))
    # Testing how much of a difference the points for basis functions make
 
#    basis.append(basis_function(Temp_sim,rho_sim,args.iRef,88.,108.,0.375,0.375,12.,12.)) 
#    basis[0].validate_ref()
#    basis.append(basis_function(Temp_sim,rho_sim,args.iRef,98.,98.,0.365,0.385,12.,12.))
#    basis[1].validate_ref()
#    basis.append(basis_function(Temp_sim,rho_sim,args.iRef,88.,108.,0.365,0.385,12.,12.))
#    basis[2].validate_ref()

if __name__ == '__main__':
    '''
    python basis_function_class.py --nRefs XX
  
    "--nRefs XX" or "--iRef XX" flag is required, sets the integer value for nRefs or iRef
    '''

    main()