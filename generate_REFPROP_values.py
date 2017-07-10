from __future__ import division
import numpy as np 
import os.path
import CoolProp.CoolProp as CP

#Before running script run, "pip install pymbar, pip install CoolProp"

compound='ETHANE'

CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH,'/home/ram9/REFPROP-cmake/build/')

Mw = CP.PropsSI('M','REFPROP::'+compound) #[kg/mol]


# Physical constants
N_A = 6.02214086e23 #[/mol]
nm3_to_ml = 10**21
nm3_to_m3 = 10**27
bar_nm3_to_kJ_per_mole = 0.0602214086
R_g = 8.3144598 / 1000. #[kJ/mol/K]

def U_to_u(U,T): #Converts internal energy into reduced potential energy in NVT ensemble
    beta = 1./(R_g*T)
    u = beta*(U)
    return u

g_start = 28 #Row where data starts in g_energy output
g_t = 0 #Column for the snapshot time
g_en = 2 #Column where the potential energy is located
g_T = 4 #Column where T is located
g_p = 5 #Column where p is located

iEpsRef = 0
iSigmaRef = 0

ITIC = np.array(['Isotherm', 'Isochore'])
Temp_ITIC = {'Isochore':[],'Isotherm':[]}
rho_ITIC = {'Isochore':[],'Isotherm':[]}
Nmol = {'Isochore':[],'Isotherm':[]}
Temps = {'Isochore':[],'Isotherm':[]}
rhos = {'Isochore':[],'Isotherm':[]}
nTemps = {'Isochore':[],'Isotherm':[]}
nrhos = {'Isochore':[],'Isotherm':[]}

Temp_sim = np.empty(0)
rho_sim = np.empty(0)
Nmol_sim = np.empty(0)

#Extract state points from ITIC files

for run_type in ITIC:

    run_type_Settings = np.loadtxt('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/'+run_type+'Settings.txt',skiprows=1)

    Nmol[run_type] = run_type_Settings[:,0]
    Lbox = run_type_Settings[:,1] #[nm]
    Temp_ITIC[run_type] = run_type_Settings[:,2] #[K]
    Vol = Lbox**3 #[nm3]
    rho_ITIC[run_type] = Nmol[run_type] / Vol #[molecules/nm3]
    rhos[run_type] = np.unique(rho_ITIC[run_type])
    nrhos[run_type] = len(rhos[run_type])
    Temps[run_type] = np.unique(Temp_ITIC[run_type])
    nTemps[run_type] = len(Temps[run_type]) 
 
    Temp_sim = np.append(Temp_sim,Temp_ITIC[run_type])
    rho_sim = np.append(rho_sim,rho_ITIC[run_type])
    Nmol_sim = np.append(Nmol_sim,Nmol[run_type])

rho_mass = rho_sim * Mw / N_A * nm3_to_m3 #[kg/m3]

#Generate REFPROP values, prints out into a file in the correct directory

def REFPROP_UP(TSim,rho_mass,NmolSim,compound,iEpsRef,iSigmaRef):
    RP_U = CP.PropsSI('UMOLAR','T',TSim,'D',rho_mass,'REFPROP::'+compound) / 1e3 #[kJ/mol]
    RP_U_ig = CP.PropsSI('UMOLAR','T',TSim,'D',0,'REFPROP::'+compound) / 1e3 #[kJ/mol]
    RP_U_dep = RP_U - RP_U_ig
    RP_U_depRT = RP_U_dep / TSim / R_g
    RP_U_depN = RP_U_dep * NmolSim
    RP_Z = CP.PropsSI('Z','T',TSim,'D',rho_mass,'REFPROP::'+compound)
    RP_P = CP.PropsSI('P','T',TSim,'D',rho_mass,'REFPROP::'+compound) / 1e5 #[bar]
    RP_Z1rho = (RP_Z - 1.)/rho_mass

    f = open('/home/ram9/Ethane/Gromacs/TraPPEfs/e'+str(iEpsRef)+'s'+str(iSigmaRef)+'/REFPROP_UPZ','w')

    for iState, Temp in enumerate(TSim):

        f.write(str(RP_U_depN[iState])+'\t')
        f.write(str(RP_P[iState])+'\t')
        f.write(str(RP_Z[iState])+'\t')
        f.write(str(RP_Z1rho[iState])+'\n')

    f.close()        

    return RP_U_depN, RP_P, RP_Z, RP_Z1rho

REFPROP_UP(Temp_sim,rho_mass,Nmol_sim,compound,iEpsRef,iSigmaRef)



