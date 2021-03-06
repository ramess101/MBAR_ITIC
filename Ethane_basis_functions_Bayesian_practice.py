from __future__ import division
import numpy as np 
import os, sys, argparse, shutil
from pymbar import MBAR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymbar import timeseries
import CoolProp.CoolProp as CP
#from REFPROP_values import *
import subprocess
import time
from scipy.optimize import minimize, minimize_scalar, fsolve
import scipy.integrate as integrate
from scipy.stats import distributions
from create_tab import *
from PCF_PSO_modules import *
from TDE_Ethane import *
from basis_function_class import basis_function, UP_basis_mult_refs
from Metropolis_Bayesian import metropolis_tuned
from VLE_model_fit import *
from TDE_Ethane import *

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

#print(fpath_all)

print(os.getcwd())
time.sleep(2)

eps_low = np.loadtxt('eps_low')
eps_guess = np.loadtxt('eps_guess')
eps_high = np.loadtxt('eps_high')
#eps_range_low = np.loadtxt('eps_range_low')
#eps_range_high = np.loadtxt('eps_range_high')
#eps_low = eps_guess*(1.-eps_range_low)
#eps_high = eps_guess*(1.+eps_range_high)

sig_low = np.loadtxt('sig_low')
sig_guess = np.loadtxt('sig_guess')
sig_high = np.loadtxt('sig_high')
#sig_range = np.loadtxt('sig_range')
#sig_low= sig_guess * (1.-sig_range)
#sig_high=sig_guess * (1.+sig_range)

lam_guess = np.loadtxt('lam_guess')
lam_low = np.loadtxt('lam_low')
lam_high = np.loadtxt('lam_high')

###

iRef = int(np.loadtxt('iRef'))
iRefmin = int(np.loadtxt('iRefmin'))

mult_refs = True

if mult_refs: #Trying to make this backwards comptabile so that it can work when only a single reference

    nRefs = iRef - iRefmin + 1 #Plus 1 because we need one more for the zeroth state
    iRefs = range(iRefmin,iRef+1)  

else:
    
    nRefs = 1
    iRefs = iRef

iRerun = 0
#iRerun = iRef+1 #RAM: I don't think this is actually necessary because typically I don't use the rerun number directly

eps_sig_lam_refs = np.empty([nRefs,3])
    
for iiiRef, iiRef in enumerate(iRefs): #We want to perform a rerun with each reference

    fpathRef = "../ref"+str(iiRef)+"/"
    eps_sig_lam_ref = np.loadtxt(fpathRef+'eps_sig_lam_ref')
    eps_sig_lam_refs[iiiRef,:] = eps_sig_lam_ref
                    
#print(eps_sig_lam_refs)

def U_to_u(U,T): #Converts internal energy into reduced potential energy in NVT ensemble
    beta = 1./(R_g*T)
    u = beta*(U)
    return u

def r_min_calc(sig, n=12., m=6.):
    r_min = (n/m*sig**(n-m))**(1./(n-m))
    return r_min

sig_TraPPE = 0.375 #[nm]
lam_TraPPE = 12
r_min_TraPPE = r_min_calc(sig_TraPPE,lam_TraPPE)

def constraint_sig(params): #If lambda is greater than 12 sigma must be greater than sigma TraPPE
    sig = params[1]
    lam = params[2]
    if lam < lam_TraPPE:
        return sig_TraPPE - sig
    elif lam >= lam_TraPPE:
        return sig - sig_TraPPE
    
def constraint_rmin(params): #If lambda is greater than 12 rmin must be less than rmin TraPPE
    sig = params[1]
    lam = params[2]
    if lam < lam_TraPPE:
        return r_min_calc(sig,lam) - r_min_TraPPE
    elif lam >= lam_TraPPE:
        return r_min_TraPPE - r_min_calc(sig,lam)

def REFPROP_UP(TSim,rho_mass,NmolSim,compound):
    RP_U = CP.PropsSI('UMOLAR','T',TSim,'D',rho_mass,'REFPROP::'+compound) / 1e3 #[kJ/mol]
    RP_U_ig = CP.PropsSI('UMOLAR','T',TSim,'D',0,'REFPROP::'+compound) / 1e3 #[kJ/mol]
    RP_U_dep = RP_U - RP_U_ig
    RP_U_depRT = RP_U_dep / TSim / R_g
    RP_U_depN = RP_U_dep * NmolSim
    RP_Z = CP.PropsSI('Z','T',TSim,'D',rho_mass,'REFPROP::'+compound)
    RP_P = CP.PropsSI('P','T',TSim,'D',rho_mass,'REFPROP::'+compound) / 1e5 #[bar]
    RP_Z1rho = (RP_Z - 1.)/rho_mass

    f = open('REFPROP_UPZ','w')

    for iState, Temp in enumerate(TSim):

        f.write(str(RP_U_depN[iState])+'\t')
        f.write(str(RP_P[iState])+'\t')
        f.write(str(RP_Z[iState])+'\t')
        f.write(str(RP_Z1rho[iState])+'\n')

    f.close()        

    return RP_U_depN, RP_P, RP_Z, RP_Z1rho

#Generate REFPROP values, prints out into a file in the correct directory

RP_U_depN, RP_P, RP_Z, RP_Z1rho = REFPROP_UP(Temp_sim,rho_mass,Nmol_sim,compound)

def objective_ITIC(eps_sig_lam,prop_type,basis_fun,PCFR_hat): 
    global iRerun
    
    if PCFR_hat:
        USim, dUSim, PSim, dPSim, ZSim = PCFR_estimates(eps_sig_lam,iRerun,PCFR_hat)
    else:
        USim, dUSim, PSim, dPSim, ZSim = MBAR_estimates(eps_sig_lam,iRerun,basis_fun)
    
    Tsat, rhoLSim, PsatSim, rhovSim = ITIC_calc(USim, ZSim)
    
    #print(Tsat)
    #print(rhoLSim)
    #print(PsatSim)
    #print(rhovSim)
    print('iRerun= '+str(iRerun))
    f = open('ITIC_'+str(iRerun),'w')
    f.write('Tsat (K)\trhoL (kg/m3)\tPsat (bar)\trhov (kg/m3)')
    for Tsatprint,rhoLprint,Psatprint,rhovprint in zip(Tsat,rhoLSim,PsatSim,rhovSim):
        f.write('\n'+str(Tsatprint))
        f.write('\t'+str(rhoLprint))
        f.write('\t'+str(Psatprint))
        f.write('\t'+str(rhovprint))
    f.close()

    if np.any(Tsat[Tsat>RP_Tmin]<RP_TC):

        RP_rhoL = CP.PropsSI('D','T',Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)],'Q',0,'REFPROP::'+compound) #[kg/m3]   
        RP_rhov = CP.PropsSI('D','T',Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)],'Q',1,'REFPROP::'+compound) #[kg/m3]
        RP_Psat = CP.PropsSI('P','T',Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)],'Q',1,'REFPROP::'+compound)/100000. #[bar]

        devrhoL = rhoLSim[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)] - RP_rhoL #In case Tsat is greater than RP_TC
        devPsat = PsatSim[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)] - RP_Psat
        devrhov = rhovSim[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)] - RP_rhov

        w8_rhoL = RP_rhoL*TDE_rel_rhol_hat(Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)])
        w8_Psat = RP_Psat*TDE_rel_Psat_hat(Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)])
                     
        devrhoL /= w8_rhoL
        devPsat /= w8_Psat

        devU = USim - RP_U_depN
        devP = PSim - RP_P
        devZ = ZSim - RP_Z

    else:

        devrhoL = 1e8
        devPsat = 1e8
        devrhov = 1e8
        devU = 1e8
        devP = 1e8
        devZ = 1e8
   
    SSErhoL = np.sum(np.power(devrhoL,2))
    SSEPsat = np.sum(np.power(devPsat,2)) 
    SSErhov = np.sum(np.power(devrhov,2)) 
    SSEU = np.sum(np.power(devU,2))
    SSEP = np.sum(np.power(devP,2))
    SSEZ = np.sum(np.power(devZ,2))
    
    SSE = 0
    
    for prop in prop_type:
        if prop == 'rhoL':
            SSE += SSErhoL
        elif prop == 'Psat':
            SSE += SSEPsat
        elif prop == 'rhov':
            SSE += SSErhov
        elif prop == 'U':
            SSE += SSEU
        elif prop == 'P':
            SSE += SSEP
        elif prop == 'Z':
            SSE += SSEZ
    
    f = open('F_ITIC_all','a')
    f.write('\n'+str(SSE))
    f.close()
    
    f = open('SSE_rhoL_all','a')
    f.write('\n'+str(SSErhoL))
    f.close()
    
    f = open('SSE_Psat_all','a')
    f.write('\n'+str(SSEPsat))
    f.close()
    
    f = open('SSE_rhov_all','a')
    f.write('\n'+str(SSErhov))
    f.close()
    
    f = open('SSE_U_all','a')
    f.write('\n'+str(SSEU))
    f.close()
    
    f = open('SSE_P_all','a')
    f.write('\n'+str(SSEP))
    f.close()
    
    f = open('SSE_Z_all','a')
    f.write('\n'+str(SSEZ))
    f.close()
    
    iRerun += 1
    
    #print(RP_rhoL)
    #print(RP_Psat)
    
    return SSE#, SSE #This is the only way to get fsolve to work

def ITIC_calc(USim,ZSim):
    #global Temp_sim, rho_mass, Temp_ITIC, rhos_mass_ITIC, nrhos, Mw
    Temp_IT = Temp_ITIC['Isotherm'].astype(float)[0]
    rho_IT = rhos_mass_ITIC['Isotherm'].astype(float)
    Z1rho = ((ZSim[0:len(rho_IT)]-1.)/rho_IT).astype(float) #[m3/kg]
    
    #print(Temp_IT)
    #print(rho_IT)
    #print(Z1rho)
    
    Z1rho_hat = np.poly1d(np.polyfit(rho_IT,Z1rho,3)) #3rd or 4th order polynomial fit
    
    #Since REFPROP won't give me B2 above TC for some reason, I will simply 
    RP_Adep_IT_0 = CP.PropsSI('ALPHAR','T',Temp_IT,'D',rho_IT[0],'REFPROP::'+compound)
    
    Adep_IT = lambda rhoL: integrate.quad(Z1rho_hat,rho_IT[0],rhoL)[0] + RP_Adep_IT_0
    
    # Verifying that the curves look like they should                                     
#    import matplotlib.pyplot as plt                                     
#    rhoL_plot = np.linspace(0,600,100)
#    Adep_plot = np.zeros(len(rhoL_plot))
#    for i, rho in enumerate(rhoL_plot):
#        Adep_plot[i] = Adep_IT(rho)
#        
#    rhoL_sim = rhos_mass_ITIC['Isochore'].astype(float)
#    Adep_sim = np.zeros(len(rhoL_sim))    
#    for i, rho in enumerate(rhoL_sim):
#        Adep_sim[i] = Adep_IT(rho)
#
#    plt.plot(rhoL_plot,Adep_plot)
#    plt.plot(rhoL_sim,Adep_sim,marker='o')
#    plt.show()                                     
    
    beta_IT = 1./Temp_IT
                                       
    Tsat = np.zeros(nrhos['Isochore']) 
    Psat = np.zeros(nrhos['Isochore'])                                
    rhoL = np.zeros(nrhos['Isochore'])
    rhov = np.zeros(nrhos['Isochore'])
    Adep_IC = np.zeros(nrhos['Isochore'])
    Adep_ITIC = np.zeros(nrhos['Isochore'])
                                         
    for iIC, rho_IC in enumerate(rhos_mass_ITIC['Isochore'].astype(float)):
        Temp_IC = Temp_sim[rho_mass == rho_IC]
        U_IC = USim[rho_mass == rho_IC]
        Z_IC = ZSim[rho_mass == rho_IC]
        N_IC = Nmol_sim[rho_mass == rho_IC]
        
        beta_IC = 1./Temp_IC
                                          
        #U_IC = UT_IC * Temp_IC / 1000.
        UT_IC = U_IC / R_g/ Temp_IC/ N_IC
        #UT_IC = U_IC * N_IC
        
        #print(Temp_IC)
        #print(U_IC)
        #print(Z_IC)
        #print(beta_IC)
        #print(UT_IC)
        #plt.scatter(beta_IC,UT_IC,label=rho_IC)
        #plt.legend()
        #plt.show()
        
        ### Attempt to avoid relying on REFPROP ZL
        conv_ZL = False
        Z_L = 0.
        iterations = 0
        TOL_ZL = 1e-6
        max_IT = 10
        while not conv_ZL:
            
            p_Z_IC = np.polyfit(beta_IC,Z_IC-Z_L,2)
            if p_Z_IC[0] > 0. or p_Z_IC[1]**2. - 4.*p_Z_IC[0]*p_Z_IC[2] < 0.: # If the concavity is not correct then just use a linear fit since it should be concave down. Also, if no root then this is problematic.
                p_Z_IC = np.polyfit(beta_IC,Z_IC-Z_L,1)
            p_UT_IC = np.polyfit(beta_IC,UT_IC,1)
            Z_IC_hat = np.poly1d(p_Z_IC)+Z_L
            UT_IC_hat = np.poly1d(p_UT_IC)
            U_IC_hat = lambda beta: UT_IC_hat(beta)/beta                                     
            
            beta_sat = np.roots(p_Z_IC).max() #We want the positive root, this has problems when concave up, so I included an if statement above (alternatively could use if statement here with .min)
            
            #print(U_IC_hat(beta_IT))
            #print(beta_IT)
            #print(beta_sat)
                               
            Adep_IC[iIC] = integrate.quad(U_IC_hat,beta_IT,beta_sat)[0]
            Adep_ITIC[iIC] = Adep_IT(rho_IC) + Adep_IC[iIC]
            
            #print(Adep_IT(rho_IC))
            #print(beta_sat)
            #print(Adep_IC)
            #print(Adep_ITIC)
            
            Z_L = Z_IC_hat(beta_sat) # Should be 0 for first iteration
#            print('Z_L = '+str(Z_L))
            Tsat[iIC] = 1./beta_sat
            rhoL[iIC] = rho_IC
                
            #print(Tsat)
            #print(rhoL)
            #print(Psat)
            
            if Tsat[iIC] > RP_Tmin and Tsat[iIC] < RP_TC:
                B2 = CP.PropsSI('BVIRIAL','T',Tsat[iIC],'Q',1,'REFPROP::'+compound) #[m3/mol]
                B2 /= Mw #[m3/kg]
                B3 = CP.PropsSI('CVIRIAL','T',Tsat[iIC],'Q',1,'REFPROP::'+compound) #[m3/mol]2
                B3 /= Mw**2 #[m3/kg]
            else:
                B2 = 0.
                B3 = 0.
            eq2_14 = lambda(rhov): Adep_ITIC[iIC] + Z_L - 1 + np.log(rhoL[iIC]/rhov) - 2*B2*rhov + 1.5*B3*rhov**2
            eq2_15 = lambda(rhov): rhov - rhoL[iIC]*np.exp(Adep_ITIC[iIC] + Z_L - 1 - 2*B2*rhov - 1.5*B3*rhov**2)               
            SE = lambda rhov: (eq2_15(rhov) - 0.)**2
            guess = (0.1,)
            rho_c_RP = CP.PropsSI('RHOCRIT','REFPROP::'+compound)
            bnds = ((0., rho_c_RP),)
            opt = minimize(SE,guess,bounds=bnds)
            rhov[iIC] = opt.x[0] #[kg/m3]
            
            Zv = (1. + B2*rhov[iIC] + B3*rhov[iIC]**2)
            Psat[iIC] = Zv * rhov[iIC] * R_g * Tsat[iIC] / Mw #[kPa]
            Psat[iIC] /= 100. #[bar]
            
            #Z_L_it = Psat[iIC]*100./rhoL[iIC]/R_g/Tsat[iIC]*Mw
            Z_L_it = Zv * rhov[iIC]/rhoL[iIC] #Simpler to just scale with saturated vapor since same pressure and temperature
            
            if np.abs(Z_L - Z_L_it) < TOL_ZL or iterations > max_IT:
                conv_ZL = True
            
            Z_L = Z_L_it
            iterations += 1
#            print('Z_L_it = '+str(Z_L_it))
    
    #plt.plot(rhoL,Adep_IC,label='IC')
    #plt.plot(rhoL,Adep_ITIC,label='ITIC')
    #plt.legend()
    #plt.show()
    
    #print(Adep_IC)
    #print(Adep_ITIC)
    
    return Tsat, rhoL, Psat, rhov

def rerun_refs():
    
    for iiRef, eps_sig_lam in enumerate(eps_sig_lam_refs): #We want to perform a rerun with each reference

        fpathRef = "../ref"+str(iiRef)+"/"
        print(fpathRef)
    
        for iiiRef, eps_sig_lam in enumerate(eps_sig_lam_refs):
    
            f = open(fpathRef+'eps_it','w')
            f.write(str(eps_sig_lam[0]))
            f.close()
        
            f = open(fpathRef+'sig_it','w')
            f.write(str(eps_sig_lam[1]))
            f.close()
        
            f = open(fpathRef+'lam_it','w')
            f.write(str(eps_sig_lam[2]))
            f.close()
            
            f = open(fpathRef+'iRerun','w')
            f.write(str(iiiRef))
            f.close()
        
            subprocess.call(fpathRef+"EthaneRerunITIC_subprocess")

#print(objective_ITIC(1.))

def calc_Deltaf(eps_sig_lam,iRerun,basis_fun):
       
    f = open('eps_all','a')
    f.write('\n'+str(eps_sig_lam[0]))
    f.close()
    
    f = open('sig_all','a')
    f.write('\n'+str(eps_sig_lam[1]))
    f.close()
    
    f = open('lam_all','a')
    f.write('\n'+str(eps_sig_lam[2]))
    f.close()
    
    iSets = [int(iRerun)]*(nRefs + 1) #Plus 1 because we need one more for the rerun  
    
    for iiiRef, iiRef in enumerate(iRefs): #We want to perform a rerun with each reference
    
        fpathRef = "../ref"+str(iiRef)+"/"
        #print(fpathRef)
    
        f = open(fpathRef+'eps_it','w')
        f.write(str(eps_sig_lam[0]))
        f.close()
    
        f = open(fpathRef+'sig_it','w')
        f.write(str(eps_sig_lam[1]))
        f.close()
    
        f = open(fpathRef+'lam_it','w')
        f.write(str(eps_sig_lam[2]))
        f.close()
        
        f = open(fpathRef+'iRerun','w')
        f.write(str(iRerun))
        f.close()
        
        iSets[iiiRef] = iiRef
             
    print('Calculating for epsilon = '+str(eps_sig_lam[0])+' sigma = '+str(eps_sig_lam[1])+' lambda = '+str(eps_sig_lam[2]))

    f = open('eps_sig_lam_all','a')
    f.write(str(eps_sig_lam[0])+'\t')
    f.write(str(eps_sig_lam[1])+'\t')
    f.write(str(eps_sig_lam[2])+'\n')
    f.close()
                              
    LJ_total_basis_refs, U_total_basis_refs, press_basis_refs = UP_basis_mult_refs(basis_fun)
        
    for iiiRef, iiRef in enumerate(iRefs):
        
        LJ_temp, U_temp, press_temp = basis_fun[iiiRef].UP_basis_states(eps_sig_lam)
        
        if iiiRef == 0:
        
            nSnaps = LJ_temp.shape[1]
            LJ_total_eps_sig_lam = np.zeros([nRefs,nStates,nSnaps])
            U_total_eps_sig_lam = np.zeros([nRefs,nStates,nSnaps])
            press_eps_sig_lam = np.zeros([nRefs,nStates,nSnaps])
            
        LJ_total_eps_sig_lam[iiiRef], U_total_eps_sig_lam[iiiRef], press_eps_sig_lam[iiiRef] = LJ_temp, U_temp, press_temp         
             
    nSets = len(iSets)
    
    N_k = [0]*nSets #This makes a list of nSets elements, need to be 0 for MBAR to work
        
    # Analyze snapshots
    
    U_MBAR = np.zeros([nStates,nSets])
    dU_MBAR = np.zeros([nStates,nSets])
    P_MBAR = np.zeros([nStates,nSets])
    dP_MBAR = np.zeros([nStates,nSets])
    Z_MBAR = np.zeros([nStates,nSets])
    Z1rho_MBAR = np.zeros([nStates,nSets])
    Neff_MBAR = np.zeros([nStates,nSets])
    Deltaf_MBAR = np.zeros([nStates,nSets])
    #print(nTemps['Isochore'])
    
    for iState in range(nStates):
        
#        rho_state = rho_sim[iState]
#        Nstate = Nmol_sim[iState]
#        fpath = fpath_all[iState]
                                    
        for iiiRef, iiRef in enumerate(iRefs): # To initialize arrays we must know how many snapshots come from each reference
            
            nSnapsRef = len(LJ_total_basis_refs[iiiRef,0,iState,:])#Number of snapshots
            #print(nSnapsRef)
            
            N_k[iiiRef] = nSnapsRef # Previously this was N_k[iter] because the first iSet was set as 0 no matter what. Now we use 'Ref' so we want to use iSet here and iter for identifying
           
        nSnaps = np.sum(N_k)
        #print(N_k)
        #print(nSnaps)
        
        press = np.zeros([nSets,nSnaps])
        U_total = np.zeros([nSets,nSnaps])
        LJ_total = np.zeros([nSets,nSnaps])

        for iSet, enum in enumerate(iSets):
#            print(iSet)
#            print(enum)
            frame_shift = 0
            
            for iiiRef, iiRef in enumerate(iRefs):
#                print(iiiRef,iiRef)
                if enum > (nRefs-1):
                    LJ_total_basis_rr, U_total_basis_rr, press_basis_rr = LJ_total_eps_sig_lam[iiiRef,iState], U_total_eps_sig_lam[iiiRef,iState], press_eps_sig_lam[iiiRef,iState]
                else:    
                    LJ_total_basis_rr, U_total_basis_rr, press_basis_rr = LJ_total_basis_refs[iiiRef,enum,iState], U_total_basis_refs[iiiRef,enum,iState], press_basis_refs[iiiRef,enum,iState]
                
#                print(LJ_total_basis_rr.shape)         
                
                nSnapsRef = N_k[iiiRef]
                #print(nSnapsRef)
                assert nSnapsRef == LJ_total_basis_rr.shape[0], 'The value of N_k does not match the length of the energy file for iState='+str(iState)+', iSet='+str(iSet)+', and iiRef='+str(iiRef)
                
                for frame in xrange(nSnapsRef):
                    LJ_total[iSet][frame+frame_shift] = LJ_total_basis_rr[frame]
                    U_total[iSet][frame+frame_shift] = U_total_basis_rr[frame]
                    press[iSet][frame+frame_shift] = press_basis_rr[frame]
                             
                frame_shift += nSnapsRef
        
        u_total = U_to_u(U_total,Temp_sim[iState]) #Call function to convert U to u
           
        u_kn = u_total
        
        mbar = MBAR(u_kn,N_k)
        
        (Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)
        #print(Deltaf_ij)
        #print(Deltaf_ij.shape)
        Deltaf_MBAR[iState] = Deltaf_ij[0,:]    
    return Deltaf_MBAR

def MBAR_estimates(eps_sig_lam,iRerun,basis_fun,f_ki_loaded):
    
    #eps = eps.tolist()
    
    f = open('eps_all','a')
    f.write('\n'+str(eps_sig_lam[0]))
    f.close()
    
    f = open('sig_all','a')
    f.write('\n'+str(eps_sig_lam[1]))
    f.close()
    
    f = open('lam_all','a')
    f.write('\n'+str(eps_sig_lam[2]))
    f.close()
    
    iSets = [int(iRerun)]*(nRefs + 1) #Plus 1 because we need one more for the rerun  
    
    for iiiRef, iiRef in enumerate(iRefs): #We want to perform a rerun with each reference
    
        fpathRef = "../ref"+str(iiRef)+"/"
        #print(fpathRef)
    
        f = open(fpathRef+'eps_it','w')
        f.write(str(eps_sig_lam[0]))
        f.close()
    
        f = open(fpathRef+'sig_it','w')
        f.write(str(eps_sig_lam[1]))
        f.close()
    
        f = open(fpathRef+'lam_it','w')
        f.write(str(eps_sig_lam[2]))
        f.close()
        
        f = open(fpathRef+'iRerun','w')
        f.write(str(iRerun))
        f.close()
        
        iSets[iiiRef] = iiRef
             
    print('Calculating for epsilon = '+str(eps_sig_lam[0])+' sigma = '+str(eps_sig_lam[1])+' lambda = '+str(eps_sig_lam[2]))

    f = open('eps_sig_lam_all','a')
    f.write(str(eps_sig_lam[0])+'\t')
    f.write(str(eps_sig_lam[1])+'\t')
    f.write(str(eps_sig_lam[2])+'\n')
    f.close()
             
    #print('iRerun='+str(iRerun))
    #print('iSets='+str(iSets))
    #print('nRefs='+str(nRefs))
                 
    LJ_total_basis_refs, U_total_basis_refs, press_basis_refs = UP_basis_mult_refs(basis_fun)
        
    for iiiRef, iiRef in enumerate(iRefs):
        
        LJ_temp, U_temp, press_temp = basis_fun[iiiRef].UP_basis_states(eps_sig_lam)
        
        if iiiRef == 0:
        
            nSnaps = LJ_temp.shape[1]
            LJ_total_eps_sig_lam = np.zeros([nRefs,nStates,nSnaps])
            U_total_eps_sig_lam = np.zeros([nRefs,nStates,nSnaps])
            press_eps_sig_lam = np.zeros([nRefs,nStates,nSnaps])
            
        LJ_total_eps_sig_lam[iiiRef], U_total_eps_sig_lam[iiiRef], press_eps_sig_lam[iiiRef] = LJ_temp, U_temp, press_temp         
             
    nSets = len(iSets)
    
    N_k = [0]*nSets #This makes a list of nSets elements, need to be 0 for MBAR to work
        
    # Analyze snapshots
    
    U_MBAR = np.zeros([nStates,nSets])
    dU_MBAR = np.zeros([nStates,nSets])
    P_MBAR = np.zeros([nStates,nSets])
    dP_MBAR = np.zeros([nStates,nSets])
    Z_MBAR = np.zeros([nStates,nSets])
    Z1rho_MBAR = np.zeros([nStates,nSets])
    Neff_MBAR = np.zeros([nStates,nSets])
    #print(nTemps['Isochore'])
    
    for iState in range(nStates):
        
#        rho_state = rho_sim[iState]
#        Nstate = Nmol_sim[iState]
#        fpath = fpath_all[iState]
                                    
        for iiiRef, iiRef in enumerate(iRefs): # To initialize arrays we must know how many snapshots come from each reference
            
            nSnapsRef = len(LJ_total_basis_refs[iiiRef,0,iState,:])#Number of snapshots
            #print(nSnapsRef)
            
            N_k[iiiRef] = nSnapsRef # Previously this was N_k[iter] because the first iSet was set as 0 no matter what. Now we use 'Ref' so we want to use iSet here and iter for identifying
           
        nSnaps = np.sum(N_k)
        #print(N_k)
        #print(nSnaps)
        
        press = np.zeros([nSets,nSnaps])
        U_total = np.zeros([nSets,nSnaps])
        LJ_total = np.zeros([nSets,nSnaps])

        for iSet, enum in enumerate(iSets):
            #print(iSet)
            #print(enum)
            frame_shift = 0
            
            for iiiRef, iiRef in enumerate(iRefs):
                #print(iiiRef,iiRef)
                #if enum > (nRefs-1): #RAM This had problems when iRefs[0] != 0
                if iSet == nRefs+1-1: #RAM This should be more robust, we will only ever have one rerun that is not a reference. I included +1-1 to make it clear that it would be nRefs+1 if the index started at 1, but we need to subtract 1 for zero point index
                    LJ_total_basis_rr, U_total_basis_rr, press_basis_rr = LJ_total_eps_sig_lam[iiiRef,iState], U_total_eps_sig_lam[iiiRef,iState], press_eps_sig_lam[iiiRef,iState]
                else: 
                    #LJ_total_basis_rr, U_total_basis_rr, press_basis_rr = LJ_total_basis_refs[iiiRef,enum,iState], U_total_basis_refs[iiiRef,enum,iState], press_basis_refs[iiiRef,enum,iState] #RAM: I don't think we want this to be enum anymore because iSets does not start at 0 
                    LJ_total_basis_rr, U_total_basis_rr, press_basis_rr = LJ_total_basis_refs[iiiRef,iSet,iState], U_total_basis_refs[iiiRef,iSet,iState], press_basis_refs[iiiRef,iSet,iState]
                
#                print(LJ_total_basis_rr.shape)         
                
                nSnapsRef = N_k[iiiRef]
                #print(nSnapsRef)
                assert nSnapsRef == LJ_total_basis_rr.shape[0], 'The value of N_k does not match the length of the energy file for iState='+str(iState)+', iSet='+str(iSet)+', and iiRef='+str(iiRef)
                
                for frame in xrange(nSnapsRef):
                    LJ_total[iSet][frame+frame_shift] = LJ_total_basis_rr[frame]
                    U_total[iSet][frame+frame_shift] = U_total_basis_rr[frame]
                    press[iSet][frame+frame_shift] = press_basis_rr[frame]
                             
                frame_shift += nSnapsRef
        
        u_total = U_to_u(U_total,Temp_sim[iState]) #Call function to convert U to u
           
        u_kn = u_total

        #print(u_kn)
        #print(f_ki_loaded[iState])
        #mbar = MBAR(u_kn,N_k,verbose=True)
        #mbar = MBAR(u_kn,N_k,verbose=True,initial_f_k = f_ki_loaded[iState])
        mbar = MBAR(u_kn,N_k,initial_f_k = f_ki_loaded[iState])

        (Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)
        #print "effective sample numbers"
        #print(Deltaf_ij)
        #print mbar.computeEffectiveSampleNumber() #Check to see if sampled adequately
        
        # MRS: The observable we are interested in is U, internal energy.  The
        # question is, WHICH internal energy.  We are interested in the
        # internal energy generated from the ith potential.  So there are
        # actually _three_ observables.
        
        # Now, the confusing thing, we can estimate the expectation of the
        # three observables in three different states. We can estimate the
        # observable of U_0 in states 0, 1 and 2, the observable of U_1 in
        # states 0, 1, and 2, etc.
                    
        EUk = np.zeros([nSets,nSets])
        dEUk = np.zeros([nSets,nSets])
        EUkn = U_total #First expectation value is internal energy
        EPk = np.zeros([nSets,nSets])
        dEPk = np.zeros([nSets,nSets])
        EPkn = press #Second expectation value is pressure
                   
        for iSet, enum in enumerate(iSets):
            
            (EUk[:,iSet], dEUk[:,iSet]) = mbar.computeExpectations(EUkn[iSet]) # potential energy of 0, estimated in state 0:2 (sampled from just 0)
            (EPk[:,iSet], dEPk[:,iSet]) = mbar.computeExpectations(EPkn[iSet]) # pressure of 0, estimated in state 0:2 (sampled from just 0)
    
            #f = open('W_'+str(iSet),'w')
            #for frame in xrange(nSnaps):
                #f.write(str(mbar.W_nk[frame,iSet])+'\n')
            #f.close()
        
            # MRS: Some of these are of no practical importance.  We are most
            # interested in the observable of U_0 in the 0th state, U_1 in the 1st
            # state, etc., or the diagonal of the matrix EA (EUk, EPk).
            U_MBAR[iState] = EUk.diagonal()
            dU_MBAR[iState] = dEUk.diagonal()
            P_MBAR[iState] = EPk.diagonal()
            dP_MBAR[iState] = dEPk.diagonal()
            Z_MBAR[iState] = P_MBAR[iState]/rho_sim[iState]/Temp_sim[iState]/R_g * bar_nm3_to_kJ_per_mole #EP [bar] rho_sim [1/nm3] Temp_sim [K] R_g [kJ/mol/K] #There is probably a better way to assign Z_MBAR
            Z1rho_MBAR[iState] = (Z_MBAR[iState] - 1.)/rho_mass[iState] * 1000. #[ml/gm]
            Neff_MBAR[iState] = mbar.computeEffectiveSampleNumber()
            
#Z_MBAR = P_MBAR/rho_sim/Temp_sim/R_g * bar_nm3_to_kJ_per_mole #EP [bar] rho_sim [1/nm3] Temp_sim [K] R_g [kJ/mol/K] #Unclear how to assing Z_MBAR without having it inside loops

                    
        #print 'Expectation values for internal energy in kJ/mol:'
        #print(U_MBAR)
        #print 'MBAR estimates of uncertainty in internal energy in kJ/mol:'
        #print(dU_MBAR)
        #print 'Expectation values for pressure in bar:'
        #print(P_MBAR)
        #print 'MBAR estimates of uncertainty in pressure in bar:'
        #print(dP_MBAR)
    
    U_rerun = U_MBAR[:,-1]
    dU_rerun = dU_MBAR[:,-1]
    P_rerun = P_MBAR[:,-1]
    dP_rerun = dP_MBAR[:,-1]
    Z_rerun = Z_MBAR[:,-1]
    Z1rho_rerun = Z1rho_MBAR[:,-1]
    Neff_rerun = Neff_MBAR[:,-1]
    #print(U_rerun)
    #print(dU_rerun)
    #print(P_rerun)
    #print(dP_rerun)
    
#    for iSet, enum in enumerate(iSets):
#    
#        f = open('MBAR_ref'+str(iRef)+'rr'+str(enum),'w')
#        
#        for iState in range(nStates):
#            
#            f.write(str(U_MBAR[iState][iSet])+'\t')
#            f.write(str(dU_MBAR[iState][iSet])+'\t')
#            f.write(str(P_MBAR[iState][iSet])+'\t')
#            f.write(str(dP_MBAR[iState][iSet])+'\t')
#            f.write(str(Z_MBAR[iState][iSet])+'\t')
#            f.write(str(Z1rho_MBAR[iState][iSet])+'\t')
#            f.write(str(Neff_MBAR[iState][iSet])+'\n')
#    
#        f.close()
    
    return U_rerun, dU_rerun, P_rerun, dP_rerun, Z_rerun,Z1rho_rerun,Neff_rerun

def compile_PCFs(iRef):
    
    fpathRef = "../ref"+str(iRef)+"/"
    print(fpathRef)
    
    g_start = 27 # Skipping the r = 0 bin
    g_PCF = 1 #The column where the PCF is located
    nbins = len(r)
    
    PCF_all = np.zeros([nbins,nStates])
    
    for iState in range(nStates):
        
        #rho_state = rho_sim[iState]
        #Nstate = Nmol_sim[iState]
        fpath = fpath_all[iState]
        
        r_PCF = open(fpathRef+fpath+'nvt_prod_rdf.xvg','r').readlines()[g_start:g_start+nbins] #Read only the PCF column
    
        for ibin in range(nbins):
            PCF_all[ibin,iState] = float(r_PCF[ibin].split()[g_PCF])
            
    # Store the PCF compilation into a single file
    
    f = open(fpathRef+'PCFs_ref'+str(iRef),'w')
    
    for ibin in range(nbins):
        
        for iState in range(nStates):
        
            f.write(str(PCF_all[ibin,iState])+'\t')
            
        else: # For the last loop of states use a new line
        
            f.write('\n')
            
    f.close()

def create_PCFRref(iRef):
    fpathRef = "../ref"+str(iRef)+"/"
    print(fpathRef)
    
    eps_sig_lam_ref = np.loadtxt(fpathRef+'eps_sig_lam_ref')
    
#    g_start = 27 # Skipping the r = 0 bin
#    nbins = len(r)
#    
#    PCF_all = np.zeros([nbins,nStates])
#    
#    for iState in range(nStates):
#        
#        #rho_state = rho_sim[iState]
#        #Nstate = Nmol_sim[iState]
#        fpath = fpath_all[iState]
#        
#        PCF_all[:,iState] = open(fpathRef+fpath+'nvt_prod_rdf.xvg','r').readlines()[g_start:g_start+nbins,1] #Read only the PCF column
    
    PCF_all = np.loadtxt(fpathRef+'PCFs_ref'+str(iRef))
                
    LJref = Mie(r,PCF_all, rho_sim, Nmol_sim, Temp_sim, eps_sig_lam_ref[0], eps_sig_lam_ref[1], eps_sig_lam_ref[2])
    Udev = 0#U_L_highP_ens - LJref.calc_Ureal()
    Pdev = 0#Pref_ens - LJref.calc_Preal()
    LJref = Mie(r,PCF_all, rho_sim, Nmol_sim, Temp_sim, eps_sig_lam_ref[0], eps_sig_lam_ref[1], eps_sig_lam_ref[2], ref=LJref,devU=Udev,devP=Pdev) #Redefine the reference system
    return LJref

def create_PCFR_hat(PCFRref):
    PCF = PCFRref.RDF
    rho = PCFRref.rho
    Temp = PCFRref.Temp
    Udev = PCFRref.devU
    Pdev = PCFRref.devP
    Mie_hat = lambda eps_sig_lam: Mie(r,PCF,rho, Nmol_sim, Temp, eps_sig_lam[0], eps_sig_lam[1], eps_sig_lam[2], ref=PCFRref,devU=Udev,devP=Pdev)
    return Mie_hat

def PCFR_estimates(eps_sig_lam,iRerun,PCFR_hat):
      
    f = open('eps_all','a')
    f.write('\n'+str(eps_sig_lam[0]))
    f.close()
    
    f = open('sig_all','a')
    f.write('\n'+str(eps_sig_lam[1]))
    f.close()
    
    f = open('lam_all','a')
    f.write('\n'+str(eps_sig_lam[2]))
    f.close()
    
    print('Calculating for epsilon = '+str(eps_sig_lam[0])+' sigma = '+str(eps_sig_lam[1])+' lambda = '+str(eps_sig_lam[2]))
    
    f = open('eps_sig_lam_all','a')
    f.write(str(eps_sig_lam[0])+'\t')
    f.write(str(eps_sig_lam[1])+'\t')
    f.write(str(eps_sig_lam[2])+'\n')
    f.close()
    
    f = open('iRerun','w')
    f.write(str(iRerun))
    f.close()
    
    PCFR_eps_sig_lam = PCFR_hat(eps_sig_lam)
    U_PCFR = PCFR_eps_sig_lam.calc_Ureal('PMF')
    dU_PCFR = U_PCFR * 0.
    P_PCFR = PCFR_eps_sig_lam.calc_Preal('')
    dP_PCFR = P_PCFR * 0.
    Z_PCFR = PCFR_eps_sig_lam.calc_Z('')
    Z1rho_PCFR = PCFR_eps_sig_lam.calc_Z1rho('')
    Neff_PCFR = U_PCFR/U_PCFR * 1001.

    U_PCFR = PCFR_eps_sig_lam.ref.calc_Ureal('') + PCFR_eps_sig_lam.calc_Ureal('zeroth') - PCFR_eps_sig_lam.ref.calc_Ureal('zeroth') #RAMtemp
    Z_PCFR = PCFR_eps_sig_lam.ref.calc_Z('') + PCFR_eps_sig_lam.calc_Z('zeroth') - PCFR_eps_sig_lam.ref.calc_Z('zeroth') #RAMtemp
           
        # The question is where the store the reference values, RDFs, and ensemble averages.
        # I can read in the snapshots for the reference and calculate the ensemble averages. Or find a way to read from the logfiles.
        # For the MBAR approach I should also store the snapshots for the previous references as an object of a class so that I don't
        # have to keep opening up files, etc.     

    f = open('PCFR_ref'+str(iRef)+'rr'+str(iRerun),'w')
    
    for iState in range(nStates):
        
        f.write(str(U_PCFR[iState])+'\t')
        f.write(str(dU_PCFR[iState])+'\t')
        f.write(str(P_PCFR[iState])+'\t')
        f.write(str(dP_PCFR[iState])+'\t')
        f.write(str(Z_PCFR[iState])+'\t')
        f.write(str(Z1rho_PCFR[iState])+'\t')
        f.write(str(Neff_PCFR[iState])+'\n')

    f.close()
    
    return U_PCFR, dU_PCFR, P_PCFR, dP_PCFR, Z_PCFR

dnorm = distributions.norm.logpdf
dgamma = distributions.gamma.logpdf

#Tfit = np.array([137.,174.,207.,236.])
Tfit = np.linspace(137,260,30)

RP_rhoL = CP.PropsSI('D','T',Tfit,'Q',0,'REFPROP::'+compound) #[kg/m3]   
RP_Psat = CP.PropsSI('P','T',Tfit,'Q',1,'REFPROP::'+compound)/100. #[kPa]

# From TRC database
TRC_data_rhol = np.loadtxt('/home/ram9/Elliott/Ethane_basis/TRC_data_rhoL.txt')
T_rhol_data = TRC_data_rhol[:,0]
rhol_data = TRC_data_rhol[:,1]

TRC_data_Psat = np.loadtxt('/home/ram9/Elliott/Ethane_basis/TRC_data_Pv.txt')
T_Psat_data = TRC_data_Psat[:,0]
Psat_data = TRC_data_Psat[:,1]

# Limit data to the range of REFPROP Tsat for isochores, could probably expand this range if needed
rhol_data = rhol_data[(T_rhol_data<260) & (T_rhol_data>137)]
T_rhol_data = T_rhol_data[(T_rhol_data<260) & (T_rhol_data>137)]

Psat_data = Psat_data[(T_Psat_data<260) & (T_Psat_data>137)]
T_Psat_data = T_Psat_data[(T_Psat_data<260) & (T_Psat_data>137)]

pu_rhol_data = TDE_rel_rhol_hat(T_rhol_data)
pu_Psat_data = TDE_rel_Psat_hat(T_Psat_data)

u_rhol_data = rhol_data*pu_rhol_data/100
u_Psat_data = Psat_data*pu_Psat_data/100

# Can just directly expand the TDE uncertainties
#pu_rhol_data *= 10.
#pu_Psat_data *= 5.
#u_rhol_data *= 10.
#u_Psat_data *= 5.                             
    
sd_rhol_data = u_rhol_data/2.
sd_Psat_data = u_Psat_data/2.
 
# Combining uncertainty for the simulations themselves    
pu_rhol_sim = 0.01*0 
pu_Psat_sim = 0.03*0
    
u_rhol_sim = rhol_data*pu_rhol_sim
u_Psat_sim = Psat_data*pu_Psat_sim
   
sd_rhol_sim = u_rhol_sim/2.
sd_Psat_sim = u_Psat_sim/2.
    
sd_rhol = np.sqrt(sd_rhol_data**2 + sd_rhol_sim**2)
sd_Psat = np.sqrt(sd_Psat_data**2 + sd_Psat_sim**2)
    
u_rhol = 2.*sd_rhol
u_Psat = 2.*sd_Psat
    
pu_rhol = u_rhol/rhol_data*100.
pu_Psat = u_Psat/Psat_data*100.

t_rhol = np.sqrt(1./sd_rhol)
t_Psat = np.sqrt(1./sd_Psat)                                   

#t_rhol = 0.3
#t_Psat = 0.03
        
def calc_posterior(eps, sig, basis_fun,f_ki_loaded=None,verbose=False):
    global iRerun

    if eps_low < eps < 1.1*eps_high and sig_low < sig < sig_high:
        
        eps_sig_lam = np.array([eps,sig,lam_guess])
        
        USim, dUSim, PSim, dPSim, ZSim, Z1rhoSim, NeffSim = MBAR_estimates(eps_sig_lam,iRerun,basis_fun,f_ki_loaded)
        
        Tsat, rhol, Psat, rhov = ITIC_calc(USim, ZSim)
        
        ITIC_fit = ITIC_VLE(Tsat,rhol,rhov,Psat)
        
        Tc_fit = ITIC_fit.Tc
        
        if Tc_fit > np.max(Tfit):
            
            # When using REFPROP correlation
#            rhol_fit = ITIC_fit.rholHat(Tfit)
#            Psat_fit = ITIC_fit.PsatHat(Tfit)/1000. #[kPa] converted to kPa so that units of rhol and Psat are on same scale for t
            # When using TRC data
            rhol_fit = ITIC_fit.rholHat(T_rhol_data)
            Psat_fit = ITIC_fit.PsatHat(T_Psat_data)/1000. #[kPa] converted to kPa so that units of rhol and Psat are on same scale for t                        
                                       
            # Priors on eps,sig
            logp = dnorm(eps, 115, 10.) + dnorm(sig, 0.375, 0.005)
            # Calculate property value for given eps, sig
            prop_hat_1 = rhol_fit
            prop_hat_2 = Psat_fit
            # Data likelihood
            # When using REFPROP values
#            logp += sum(dnorm(RP_rhoL, prop_hat_1, t_rhol**-2))
#            logp += sum(dnorm(RP_Psat, prop_hat_2, t_Psat**-2))
            # When using TRC data
            logp += sum(dnorm(rhol_data, prop_hat_1, t_rhol**-2))
            logp += sum(dnorm(Psat_data, prop_hat_2, t_Psat**-2))
            
            
            if verbose:
            
                f = open('MBAR_rr'+str(iRerun),'w')
                
                f.write('U (kJ/mol)\t dU (kJ/mol)\t P (bar)\t dP (bar)\t Z\t Z-1/rho (ml/gm)\t Neff')
                for Uprint,dUprint,Pprint,dPprint,Zprint,Z1rhoprint,Neffprint in zip(USim, dUSim, PSim, dPSim, ZSim, Z1rhoSim, NeffSim):
                    f.write('\n'+str(Uprint))
                    f.write('\t'+str(dUprint))
                    f.write('\t'+str(Pprint))
                    f.write('\t'+str(dPprint))
                    f.write('\t'+str(Zprint))
                    f.write('\t'+str(Z1rhoprint))
                    f.write('\t'+str(Neffprint))
                f.close()
                       
                f = open('ITIC_'+str(iRerun),'w')
                f.write('Tsat (K)\trhoL (kg/m3)\tPsat (bar)\trhov (kg/m3)')
                for Tsatprint,rhoLprint,Psatprint,rhovprint in zip(Tsat,rhol,Psat,rhov):
                    f.write('\n'+str(Tsatprint))
                    f.write('\t'+str(rhoLprint))
                    f.write('\t'+str(Psatprint))
                    f.write('\t'+str(rhovprint))
                f.close()
            
        else:
            print('TC was too low')
            logp = 1e-30
        
    else:
        logp = -1e30
        
        f = open('eps_all','a')
        f.write('\n'+str(eps))
        f.close()
        
        f = open('sig_all','a')
        f.write('\n'+str(sig))
        f.close()
        
        f = open('lam_all','a')
        f.write('\n'+str(lam_guess))
        f.close()
        
    if verbose:
        
        print('iRerun= '+str(iRerun))
        
        f = open('logp','a')
        f.write('\n'+str(logp))
        f.close()
    
    iRerun += 1
    
    return logp

def call_optimizers(opt_type,prop_type,lam_cons=lam_guess,cons_lam=True,basis_fun=None,PCFR_hat=None):
    
    objective = lambda eps_sig_lam: objective_ITIC(eps_sig_lam,prop_type,basis_fun,PCFR_hat)
    
    eps_sig_lam_guess = np.array([eps_guess,sig_guess,lam_guess])
    #print(eps_sig_lam_guess)
    
    if opt_type == 'scan':
        # For scanning the parameter space
        
        #f_guess = objective(eps_sig_lam_guess) #To call objective before running loop
        lam_sim = lam_cons
        
        f_opt = 1e20
        eps_opt = eps_guess
        sig_opt = sig_guess
        lam_opt = lam_guess
        
        f_ki_loaded = calc_Deltaf(eps_sig_lam_guess,1,basis_fun)

        objective = lambda eps, sig: calc_posterior(eps,sig,basis_fun,f_ki_loaded,verbose=True)      
         
        for iEps, eps_sim in enumerate(np.linspace(115,125,11)):
            for iSig, sig_sim in enumerate(np.linspace(0.375,0.380,11)):
                f_sim = objective(eps_sim,sig_sim)
                
                if f_sim < f_opt:
                    f_opt = f_sim
                    eps_opt = eps_sim
                    sig_opt = sig_sim 
                    lam_opt = lam_sim
    
    elif opt_type == 'Bayesian':

        f_ki_loaded = calc_Deltaf(eps_sig_lam_guess,1,basis_fun)
        #print(f_ki_loaded)
        #print(f_ki_loaded.shape)
        objective = lambda eps, sig: calc_posterior(eps,sig,basis_fun,f_ki_loaded)
               
        n_iter = 10000
        tune_for = 5000
        trace_tuned, acc_tuned = metropolis_tuned(objective, n_iter, (eps_guess,sig_guess), prop_var=[1,0.005], tune_for=tune_for)
        
        print(np.array(acc_tuned, float)/(n_iter-tune_for))
        
        samples = 500
    
        for i in range(samples):
            eps_sample, sig_sample = trace_tuned[np.random.randint(0, n_iter - tune_for)]
            calc_posterior(eps_sample,sig_sample,basis_fun,f_ki_loaded,verbose=True)
                
    elif opt_type == 'points':
        objective(eps_sig_lam_guess)
        eps_sig_lam_spec = np.array([121.25,0.3783,16.])
        objective_ITIC(eps_sig_lam_spec,prop_type,basis_fun,PCFR_hat)
        
    if 'f_opt' not in locals():
        f_opt = objective(np.array([eps_opt,sig_opt,lam_opt]))
    
    return eps_opt, sig_opt, lam_opt, f_opt 
            
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt","--optimizer",type=str,choices=['fsolve','steep','LBFGSB','leapfrog','scan','points','SLSQP','Bayesian'],help="choose which type of optimizer to use")
    parser.add_argument("-prop","--properties",type=str,nargs='+',choices=['rhoL','Psat','rhov','P','U','Z'],help="choose one or more properties to use in optimization" )
    parser.add_argument("-lam","--lam",help="Scan the lambda space incrementally",action="store_true")
    parser.add_argument("-bas","--basis",help="Develop the basis functions for integer values of lambda",action="store_true")
    parser.add_argument("-PCFR","--PCFR",help="Use pair correlation function rescaling",action="store_true")
    args = parser.parse_args()
    if args.PCFR: #Compile PCFs if using PCFR
        #compile_PCFs(iRef)
        PCFRref = create_PCFRref(iRef)
        PCFR_hat_Mie = create_PCFR_hat(PCFRref)
    else:
        PCFR_hat_Mie = None
    if args.optimizer:
        if not args.PCFR:
#            rerun_refs() #Perform the reruns for the references prior to anything
#            for eps_sig_lam in eps_sig_lam_refs:
#                objective_ITIC(eps_sig_lam,args.properties,args.basis,PCFR_hat_Mie) #Call objective for each of the references
            if args.basis: #Perform the reruns for the basis functions
                basis = []   
                for iiiRef, iiRef in enumerate(iRefs):
                    ### Test to see if basis functions have already been generated. Alternatively, I should just include this in the basis functions class. Of course, there is the possibility that the basis functions exist but are not the correct ones.
                    try:
                        basis.append(basis_function(Temp_sim,rho_sim,iiRef,iRefs,eps_low,eps_high,sig_low,sig_high,lam_low,lam_high,False))
                    except:
                        basis.append(basis_function(Temp_sim,rho_sim,iiRef,iRefs,eps_low,eps_high,sig_low,sig_high,lam_low,lam_high,True))
                    
                    #basis[iiiRef].validate_refs()  #The basis function class now performs this validation automatically  

                LJ_total_basis_refs, U_total_basis_refs, press_basis_refs = UP_basis_mult_refs(basis)           
#                basis.append(basis_function(Temp_sim,rho_sim,iRef,1,eps_low,eps_high,sig_low,sig_high,12.,12.,False)) 
#    #            basis.append(basis_function(Temp_sim,rho_sim,iRef,eps_low,eps_high,sig_low,sig_high,12.,18.)) 
#                basis[0].validate_ref()
#                
#                LJ_total_basis_refs, U_total_basis_refs, press_basis_refs = UP_basis_mult_refs(basis)
                #for eps_sig_lam in eps_sig_lam_refs:
                #    objective_ITIC(eps_sig_lam,args.properties,basis_fun=basis,PCFR_hat=PCFR_hat_Mie)
                
        if args.lam:
            lam_range = range(int(lam_low),int(lam_high)+1)
            eps_opt_range = np.zeros(len(lam_range))
            sig_opt_range = np.zeros(len(lam_range))
            lam_opt_range = np.zeros(len(lam_range))
            f_opt_range = np.zeros(len(lam_range))
            for ilam, lam_cons in enumerate(lam_range):
                eps_opt_range[ilam], sig_opt_range[ilam], lam_opt_range[ilam], f_opt_range[ilam] = call_optimizers(args.optimizer,args.properties,lam_cons,cons_lam=True,basis_fun=basis,PCFR_hat=PCFR_hat_Mie)
                assert lam_opt_range[ilam] == lam_cons, 'Optimal lambda is different than the constrained lambda value'
            iopt = f_opt_range.argmin()
            eps_opt, sig_opt, lam_opt, f_opt = eps_opt_range[iopt], sig_opt_range[iopt], lam_opt_range[iopt], f_opt_range[iopt]  
            lambda_plots(eps_opt_range,sig_opt_range,lam_opt_range,f_opt_range) #Print plots to see the dependence of eps, sig, and the objective function on lambda
        else:
            eps_opt, sig_opt, lam_opt, f_opt = call_optimizers(args.optimizer,args.properties,cons_lam=False,basis_fun=basis,PCFR_hat=PCFR_hat_Mie)
    else:
        print('Please specify an optimizer type')
        eps_opt = 0.
        sig_opt = 0.
        lam_opt = 0.

if __name__ == '__main__':
    '''
    python optimization_Mie_ITIC_multiple_refs.py --optimizer XX --properties XX --lam XX
  
    "--optimizer XX" flag is requiered, sets which optimizer to use
    "--properties XX" flag is required, sets which properties to optimize to
    "--lam XX" flag is optional, sets if lambda is a fixed parameter
    '''

    main()
