from __future__ import division
import numpy as np 
import os
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

nStates = len(Temp_sim)

#rho_mass = rho_sim * Mw / N_A * nm3_to_ml #[gm/ml]

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

###

iEpsRef = int(np.loadtxt('../iEpsref'))
iSigmaRef = int(np.loadtxt('../iSigref'))

def analyze_ITIC(iRerun): 
    
    USim, dUSim, PSim, dPSim, ZSim, Z1rhoSim = np.loadtxt('MBAR_e'+str(iEpsRef)+'s'+str(iSigmaRef)+'it'+str(iRerun),unpack=True)
    Tsat, rhoLSim, PsatSim, rhovSim = np.loadtxt('ITIC_'+str(iRerun),skiprows=1,unpack=True)
    
    #print(Tsat)
    #print(rhoLSim)
    #print(PsatSim)
    #print(rhovSim)

    RP_rhoL = CP.PropsSI('D','T',Tsat[Tsat<RP_TC],'Q',0,'REFPROP::'+compound) #[kg/m3]   
    RP_rhov = CP.PropsSI('D','T',Tsat[Tsat<RP_TC],'Q',1,'REFPROP::'+compound) #[kg/m3]
    RP_Psat = CP.PropsSI('P','T',Tsat[Tsat<RP_TC],'Q',1,'REFPROP::'+compound)/100000. #[bar]

    devrhoL = rhoLSim[Tsat<RP_TC] - RP_rhoL #In case Tsat is greater than RP_TC
    devPsat = PsatSim[Tsat<RP_TC] - RP_Psat
    devrhov = rhovSim[Tsat<RP_TC] - RP_rhov
                     
    devU = USim - RP_U_depN
    devP = PSim - RP_P
    devZ = ZSim - RP_Z
       
    RMSrhoL = np.sqrt(np.sum(np.power(devrhoL,2))/len(devrhoL))
    RMSPsat = np.sqrt(np.sum(np.power(devPsat,2))/len(devPsat)) 
    RMSrhov = np.sqrt(np.sum(np.power(devrhov,2))/len(devrhov)) 
    RMSU = np.sqrt(np.sum(np.power(devU,2))/len(devU))
    RMSP = np.sqrt(np.sum(np.power(devP,2))/len(devP))
    RMSZ = np.sqrt(np.sum(np.power(devZ,2))/len(devZ))
       
    f = open('RMS_rhoL_all','a')
    f.write('\n'+str(RMSrhoL))
    f.close()
    
    f = open('RMS_Psat_all','a')
    f.write('\n'+str(RMSPsat))
    f.close()
    
    f = open('RMS_rhov_all','a')
    f.write('\n'+str(RMSrhov))
    f.close()
    
    f = open('RMS_U','a')
    f.write('\n'+str(RMSU))
    f.close()
    
    f = open('RMS_P','a')
    f.write('\n'+str(RMSP))
    f.close()
    
    f = open('RMS_Z','a')
    f.write('\n'+str(RMSZ))
    f.close()
    
    return RMSrhoL, RMSPsat, RMSrhov, RMSU, RMSP, RMSZ

print(os.getcwd())
time.sleep(2)

f = open('RMS_rhoL_all','w')
f.close()

f = open('RMS_Psat_all','w')
f.close()

f = open('RMS_rhov_all','w')
f.close()

f = open('RMS_U','w')
f.close()

f = open('RMS_P','w')
f.close()

f = open('RMS_Z','w')
f.close()

# For scanning the parameter space

nReruns = int(np.loadtxt('iRerun'))
eps_sig_reruns = np.loadtxt('eps_Sigma_all',skiprows=1)
eps_reruns = eps_sig_reruns[:,0]
sig_reruns = eps_sig_reruns[:,1]
eps = np.unique(eps_reruns)
sig = np.unique(sig_reruns)
neps = len(eps)
nsig = len(sig)    

opt_type = 'Iterative'

if opt_type == 'Scan':

    RMSrhoL = np.empty([neps,nsig])
    RMSPsat = np.empty([neps,nsig])
    RMSrhov = np.empty([neps,nsig])
    RMSU = np.empty([neps,nsig])
    RMSP = np.empty([neps,nsig])
    RMSZ = np.empty([neps,nsig])

    iRerun = 1
    for ieps in range(neps):
        for isig in range(nsig):
            RMSrhoL[ieps,isig], RMSPsat[ieps,isig], RMSrhov[ieps,isig], RMSU[ieps,isig], RMSP[ieps,isig], RMSZ[ieps,isig] = analyze_ITIC(iRerun)
            iRerun += 1
            
    RMSrhoL = np.log10(RMSrhoL)
    RMSPsat = np.log10(RMSPsat)
    RMSrhov = np.log10(RMSrhov)
    RMSU = np.log10(RMSU)
    RMSP = np.log10(RMSP)
    RMSZ = np.log10(RMSZ)
            
    f = plt.figure()
    plt.contour(sig,eps,RMSrhoL)
    plt.ylabel('$\epsilon$ (kJ/mol)')
    plt.xlabel('$\sigma$ (nm)')
    plt.ylim([min(eps),max(eps)])
    plt.xlim([min(sig),max(sig)])
    f.savefig(compound+'_RMSrhoL.pdf')
    
    f = plt.figure()
    plt.contour(sig,eps,RMSPsat)
    plt.ylabel('$\epsilon$ (kJ/mol)')
    plt.xlabel('$\sigma$ (nm)')
    plt.ylim([min(eps),max(eps)])
    plt.xlim([min(sig),max(sig)])
    f.savefig(compound+'_RMSPsat.pdf')
    
    f = plt.figure()
    plt.contour(sig,eps,RMSrhov)
    plt.ylabel('$\epsilon$ (kJ/mol)')
    plt.xlabel('$\sigma$ (nm)')
    plt.ylim([min(eps),max(eps)])
    plt.xlim([min(sig),max(sig)])
    f.savefig(compound+'_RMSrhov.pdf')
    
    f = plt.figure()
    plt.contour(sig,eps,RMSU)
    plt.ylabel('$\epsilon$ (kJ/mol)')
    plt.xlabel('$\sigma$ (nm)')
    plt.ylim([min(eps),max(eps)])
    plt.xlim([min(sig),max(sig)])
    f.savefig(compound+'_RMSU.pdf')
    
    f = plt.figure()
    plt.contour(sig,eps,RMSP)
    plt.ylabel('$\epsilon$ (kJ/mol)')
    plt.xlabel('$\sigma$ (nm)')
    plt.ylim([min(eps),max(eps)])
    plt.xlim([min(sig),max(sig)])
    f.savefig(compound+'_RMSP.pdf')
    
    f = plt.figure()
    plt.contour(sig,eps,RMSZ)
    plt.ylabel('$\epsilon$ (kJ/mol)')
    plt.xlabel('$\sigma$ (nm)')
    plt.ylim([min(eps),max(eps)])
    plt.xlim([min(sig),max(sig)])
    f.savefig(compound+'_RMSZ.pdf')

elif opt_type == 'Iterative':
    
    RMSrhoL = np.empty(nReruns)
    RMSPsat = np.empty(nReruns)
    RMSrhov = np.empty(nReruns)
    RMSU = np.empty(nReruns)
    RMSP = np.empty(nReruns)
    RMSZ = np.empty(nReruns)
    
    for iRerun in range(nReruns):
        RMSrhoL[iRerun], RMSPsat[iRerun], RMSrhov[iRerun], RMSU[iRerun], RMSP[iRerun], RMSZ[iRerun] = analyze_ITIC(iRerun)
        
    f = plt.figure()
    plt.plot(RMSrhoL,marker='o',linestyle='none')
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    f.savefig(compound+'_RMSrhoL.pdf')
    
    f = plt.figure()
    plt.plot(RMSPsat,marker='o',linestyle='none')
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    f.savefig(compound+'_RMSPsat.pdf')
    
    f = plt.figure()
    plt.plot(RMSrhov,marker='o',linestyle='none')
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    f.savefig(compound+'_RMSrhov.pdf')
    
    f = plt.figure()
    plt.plot(RMSU,marker='o',linestyle='none')
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    f.savefig(compound+'_RMSU.pdf')
    
    f = plt.figure()
    plt.plot(RMSP,marker='o',linestyle='none')
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    f.savefig(compound+'_RMSP.pdf')
    
    f = plt.figure()
    plt.plot(RMSZ,marker='o',linestyle='none')
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    f.savefig(compound+'_RMSZ.pdf')