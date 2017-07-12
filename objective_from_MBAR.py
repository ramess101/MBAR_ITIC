from __future__ import division
import numpy as np 
import os.path
from pymbar import MBAR
#import matplotlib.pyplot as plt
from pymbar import timeseries
import CoolProp.CoolProp as CP
#from REFPROP_values import *

#Before running script run, "pip install pymbar, pip install CoolProp"

compound='ETHANE'

REFPROP_path='/home/ram9/REFPROP-cmake/build/' #Change this for a different system

CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH,REFPROP_path)

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

def REFPROP_UP(TSim,rho_mass,NmolSim,compound,iEpsRef,iSigmaRef):
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


def objective(RP_U_depN, RP_P, USim,PSim,dUSim=1.,dPSim=1.): 
    devU = (USim - RP_U_depN)/dUSim
    devP = (PSim - RP_P)/dPSim
    SSEU = np.sum(np.power(devU,2))
    SSEP = np.sum(np.power(devP,2)) 
    SSE = 0
    SSE += SSEU
    #SSE += SSEP
    #print(devU)
    #print(devP)
    return SSE

g_start = 28 #Row where data starts in g_energy output
g_t = 0 #Column for the snapshot time
g_en = 2 #Column where the potential energy is located
g_T = 4 #Column where T is located
g_p = 5 #Column where p is located

iEpsRef = int(np.loadtxt('../iEpsref'))
iSigmaRef = int(np.loadtxt('../iSigref'))
iRerun = int(np.loadtxt('iEps_iteration'))

iSets = [0, int(iRerun)]

nSets = len(iSets)

N_k = [0]*nSets #This makes a list of nSets elements, need to be 0 not 0. for MBAR to work


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

    run_type_Settings = np.loadtxt(run_type+'Settings.txt',skiprows=1)

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

nTemps['Isochore']=2 #Need to figure out how to get this without hardcoding
    
rho_mass = rho_sim * Mw / N_A * nm3_to_m3 #[kg/m3]

#Generate REFPROP values, prints out into a file in the correct directory

RP_U_depN, RP_P, RP_Z, RP_Z1rho = REFPROP_UP(Temp_sim,rho_mass,Nmol_sim,compound,iEpsRef,iSigmaRef)

###

nStates = len(Temp_sim)

#rho_mass = rho_sim * Mw / N_A * nm3_to_ml #[gm/ml]

# Analyze snapshots

U_MBAR = np.empty([nStates,nSets])
dU_MBAR = np.empty([nStates,nSets])
P_MBAR = np.empty([nStates,nSets])
dP_MBAR = np.empty([nStates,nSets])
Z_MBAR = np.empty([nStates,nSets])
Z1rho_MBAR = np.empty([nStates,nSets])

print(nTemps['Isochore'])

iState = 0

for run_type in ITIC: 

    for irho  in np.arange(0,nrhos[run_type]):

        for iTemp in np.arange(0,nTemps[run_type]):

            if run_type == 'Isochore':

                fpath = run_type+'/rho'+str(irho)+'/T'+str(iTemp)+'/NVT_eq/NVT_prod/'

            else:

                fpath = run_type+'/rho_'+str(irho)+'/NVT_eq/NVT_prod/'    

            for iSet, iter in enumerate(iSets):
    
                #f = open('p_rho'+str(irho)+'_T'+str(iTemp)+'_'+str(iEps),'w')
    
                en_p = open(fpath+'energy_press_e%ss%sit%s.xvg' %(iEpsRef,iSigmaRef,iter),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k

                nSnaps = len(en_p) #Number of snapshots
    
                if iSet == 0: #For the first loop we initialize these arrays

                    t = np.zeros([nSets,nSnaps])
                    en = np.zeros([nSets,nSnaps])
                    p = np.zeros([nSets,nSnaps])
                    U_total = np.zeros([nSets,nSnaps])
                    T = np.zeros([nSets,nSnaps])
                    N_k[iter] = nSnaps
     
                for frame in xrange(nSnaps):
                    t[iSet][frame] = float(en_p[frame].split()[g_t])
                    en[iSet][frame] = float(en_p[frame].split()[g_en])
                    p[iSet][frame]     = float(en_p[frame].split()[g_p])
                    T[iSet][frame] = float(en_p[frame].split()[g_T])
                    #f.write(str(p[iSet][frame])+'\n')

                U_total[iSet] = en[iSet] # If dispersion corrections include ' + en_dc[k]' after en[k]

                #f.close()
            
            u_total = U_to_u(U_total,Temp_sim[iState]) #Call function to convert U to u
               
            u_kn = u_total

            #print(u_kn)
            
            mbar = MBAR(u_kn,N_k)
            
            (Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)
            print "effective sample numbers"
            
            print mbar.computeEffectiveSampleNumber() #Check to see if sampled adequately
            
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
            EPkn = p #Second expectation value is pressure
                       
            for iSet, iter in enumerate(iSets):
                
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
                #Z1rho_MBAR[iState] = (Z_MBAR[iState] - 1.)/rho_mass #[ml/gm]

            iState += 1
                
    #Z_MBAR = P_MBAR/rho_sim/Temp_sim/R_g * bar_nm3_to_kJ_per_mole #EP [bar] rho_sim [1/nm3] Temp_sim [K] R_g [kJ/mol/K] #Unclear how to assing Z_MBAR without having it inside loops

                        
            #print 'Expectation values for internal energy in kJ/mol:'
            #print(U_MBAR)
            #print 'MBAR estimates of uncertainty in internal energy in kJ/mol:'
            #print(dU_MBAR)
            #print 'Expectation values for pressure in bar:'
            #print(P_MBAR)
            #print 'MBAR estimates of uncertainty in pressure in bar:'
            #print(dP_MBAR)

U_rerun = U_MBAR[:,1]
dU_rerun = dU_MBAR[:,1]
P_rerun = P_MBAR[:,1]
dP_rerun = dP_MBAR[:,1]

#print(U_rerun)
#print(dU_rerun)
#print(P_rerun)
#print(dP_rerun)

F_rerun = objective(RP_U_depN, RP_P, U_rerun,P_rerun)#,dU_rerun,dP_rerun)
#print(F_rerun)

f = open('F_it_'+str(iRerun),'w')

f.write(str(F_rerun))
f.close()

if iRerun > 1:

    F2 = np.loadtxt('F2_previous')
    F1 = np.loadtxt('F1_previous')

    if F2 < F1:

        F2 = F_rerun
        
        f = open('F2_current','w')

        f.write(str(F2))
        f.close()

    else:

        F1 = F_rerun
        
        f = open('F1_current','w')

        f.write(str(F1))
        f.close()

else:

    swap_initial = np.loadtxt('swap_initial')

    if iRerun == 0:

        if swap_initial == 0:
        
            F1 = F_rerun
            
            f = open('F1_current','w')
        
            f.write(str(F1))
            f.close()
           
        elif swap_initial == 1:
        
            F2 = F_rerun
            
            f = open('F2_current','w')
        
            f.write(str(F2))
            f.close()
        
    elif iRerun == 1:
        
        if swap_initial == 0:

            F2 = F_rerun
            
            f = open('F2_current','w')
    
            f.write(str(F2))
            f.close()
   
        elif swap_initial == 1:

            F1 = F_rerun
            
            f = open('F1_current','w')
    
            f.write(str(F1))
            f.close()



for iSet, iter in enumerate(iSets):

    f = open('MBAR_e'+str(iEpsRef)+'s'+str(iSigmaRef)+'it'+str(iter),'w')

    iState = 0

    for run_type in ITIC:

        for irho in np.arange(0,nrhos[run_type]):
  
            for iTemp in np.arange(0,nTemps[run_type]):
        
                f.write(str(U_MBAR[iState][iSet])+'\t')
                f.write(str(dU_MBAR[iState][iSet])+'\t')
                f.write(str(P_MBAR[iState][iSet])+'\t')
                f.write(str(dP_MBAR[iState][iSet])+'\t')
                f.write(str(Z_MBAR[iState][iSet])+'\t')
                f.write(str(Z1rho_MBAR[iState][iSet])+'\n')
                
                iState += 1

f.close()

conv_MBAR = np.loadtxt('conv_MBAR')

if conv_MBAR > 0:

    F_all = np.loadtxt('F_all',skiprows=1)
    F_all = np.append(F_all,F_rerun) #Account for the fact that F_all does not yet have the most recent F value

    #F_current = F_rerun
    #F_previous = np.loadtxt('F_it_'+str(iRerun-1))

    #eps_previous = np.loadtxt('eps_it_'+str(iRerun-1))
    #eps_current = np.loadtxt('eps_it_'+str(iRerun))
 
    #if F_current < F_previous:
    #    eps_optimal = eps_current
    #else:
    #    eps_optimal = eps_previous

    eps_all = np.loadtxt('eps_all',skiprows=1)
    eps_optimal = eps_all[np.argmin(F_all)]

    f = open('eps_optimal','w')

    f.write(str(eps_optimal))
    f.close()

    conv_eps = 0

    if iEpsRef > 0:
        eps_opt_previous = np.loadtxt('../e'+str(iEpsRef-1)+'s'+str(iSigmaRef)+'/eps_optimal')
        eps_opt_current = eps_optimal
        TOL_eps = np.loadtxt('TOL_eps')
        if np.abs(eps_opt_previous - eps_opt_current) < TOL_eps:
            conv_eps = 1
    
    f = open('conv_eps','w')
    f.write(str(conv_eps))
    f.close()

