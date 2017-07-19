from __future__ import division
import numpy as np 
import os
from pymbar import MBAR
#import matplotlib.pyplot as plt
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

iRerun = 0

def analyze_ITIC(iEps,iSig): 
    global iRerun
    
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
       
    SSErhoL = np.sum(np.power(devrhoL,2))
    SSEPsat = np.sum(np.power(devPsat,2)) 
    SSErhov = np.sum(np.power(devrhov,2)) 
    SSEU = np.sum(np.power(devU,2))
    SSEP = np.sum(np.power(devP,2))
    SSEZ = np.sum(np.power(devZ,2))
       
    f = open('SSE_rhoL_all','a')
    f.write('\n'+str(SSErhoL))
    f.close()
    
    f = open('SSE_Psat_all','a')
    f.write('\n'+str(SSEPsat))
    f.close()
    
    f = open('SSE_rhov_all','a')
    f.write('\n'+str(SSErhov))
    f.close()
    
    f = open('SSE_U','a')
    f.write('\n'+str(SSEU))
    f.close()
    
    f = open('SSE_P','a')
    f.write('\n'+str(SSEP))
    f.close()
    
    f = open('SSE_Z','a')
    f.write('\n'+str(SSEZ))
    f.close()
    
    iRerun += 1

print(os.getcwd())
time.sleep(2)

# For scanning the parameter space

for iEps in range(40):
    for iSig in range(40):
        analyze_ITIC(iEps,iSig)

