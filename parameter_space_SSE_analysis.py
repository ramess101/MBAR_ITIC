from __future__ import division
import numpy as np 
import os, sys, argparse
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

iEpsRef = int(np.loadtxt('iEpsref'))
iSigmaRef = int(np.loadtxt('iSigref'))

def analyze_ITIC(iRerun): 

    #Generate REFPROP values, prints out into a file in the correct directory

    RP_U_depN, RP_P, RP_Z, RP_Z1rho = REFPROP_UP(Temp_sim,rho_mass,Nmol_sim,compound)

    ###
    
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
    
    AAD = np.zeros(6)
    AAD[0] = np.sum(np.abs(devrhoL/RP_rhoL))*100./len(devrhoL)
    AAD[1] = np.sum(np.abs(devPsat/RP_Psat))*100./len(devPsat)
    AAD[2] = np.sum(np.abs(devrhov/RP_rhov))*100./len(devrhov)
    AAD[3] = np.sum(np.abs(devU/RP_U_depN))*100./len(devU)
    AAD[4] = np.sum(np.abs(devP/RP_P))*100./len(devP)
    AAD[5] = np.sum(np.abs(devZ/RP_Z))*100./len(devZ)
    
    bias = np.zeros(6)
    bias[0] = np.sum(devrhoL/RP_rhoL)*100./len(devrhoL)
    bias[1] = np.sum(devPsat/RP_Psat)*100./len(devPsat)
    bias[2] = np.sum(devrhov/RP_rhov)*100./len(devrhov)
    bias[3] = np.sum(devU/RP_U_depN)*100./len(devU)
    bias[4] = np.sum(devP/RP_P)*100./len(devP)
    bias[5] = np.sum(devZ/RP_Z)*100./len(devZ) 
              
    SSErhoL = np.sum(np.power(devrhoL,2))
    SSEPsat = np.sum(np.power(devPsat,2))
    SSErhov = np.sum(np.power(devrhov,2)) 
    SSEU = np.sum(np.power(devU,2))
    SSEP = np.sum(np.power(devP,2))
    SSEZ = np.sum(np.power(devZ,2))
    
    SSE = np.zeros(6)
    SSE[0] = SSErhoL
    SSE[1] = SSEPsat
    SSE[2] = SSErhov
    SSE[3] = SSEU
    SSE[4] = SSEP
    SSE[5] = SSEZ
       
    RMS = np.zeros(6)
    RMS[0] = np.sqrt(SSErhoL/len(devrhoL))
    RMS[1] = np.sqrt(SSEPsat/len(devPsat))
    RMS[2] = np.sqrt(SSErhov/len(devrhov))
    RMS[3] = np.sqrt(SSEU/len(devU))
    RMS[4] = np.sqrt(SSEP/len(devP))
    RMS[5] = np.sqrt(SSEZ/len(devZ)) 

    for iAAD,ibias,iSSE,iRMS in zip(AAD,bias,SSE,RMS):
           
        f = open('AAD_all','a')
        f.write(str(iAAD)+'\t')
        f.close()
        
        f = open('bias_all','a')
        f.write(str(ibias)+'\t')
        f.close()
        
        f = open('SSE_all','a')
        f.write(str(iSSE)+'\t')
        f.close()

        f = open('RMS_all','a')
        f.write(str(iRMS)+'\t')
        f.close()
        
    f = open('AAD_all','a')
    f.write('\n')
    f.close()
    
    f = open('bias_all','a')
    f.write('\n')
    f.close()
    
    f = open('SSE_all','a')
    f.write('\n')
    f.close()

    f = open('RMS_all','a')
    f.write('\n')
    f.close()    
           
#    f = open('SSE_rhoL_all','a')
#    f.write('\n'+str(SSErhoL))
#    f.close()
#    
#    f = open('SSE_Psat_all','a')
#    f.write('\n'+str(SSEPsat))
#    f.close()
#    
#    f = open('SSE_rhov_all','a')
#    f.write('\n'+str(SSErhov))
#    f.close()
#    
#    f = open('SSE_U','a')
#    f.write('\n'+str(SSEU))
#    f.close()
#    
#    f = open('SSE_P','a')
#    f.write('\n'+str(SSEP))
#    f.close()
#    
#    f = open('SSE_Z','a')
#    f.write('\n'+str(SSEZ))
#    f.close()
    
    return SSErhoL, SSEPsat, SSErhov, SSEU, SSEP, SSEZ

def initialize_files():
    
    print(os.getcwd())
    time.sleep(2)
    
    f = open('AAD_all','w')
    f.write('rhoL \t Psat \t rhov \t Udep \t P \t Z \n')
    f.close()
    
    f = open('bias_all','w')
    f.write('rhoL \t Psat \t rhov \t Udep \t P \t Z \n')
    f.close()
    
    f = open('SSE_all','w')
    f.write('rhoL \t Psat \t rhov \t Udep \t P \t Z \n')
    f.close()
    
    f = open('RMS_all','w')
    f.write('rhoL \t Psat \t rhov \t Udep \t P \t Z \n')
    f.close()
    
#    f = open('SSE_rhoL_all','w')
#    f.close()
#    
#    f = open('SSE_Psat_all','w')
#    f.close()
#    
#    f = open('SSE_rhov_all','w')
#    f.close()
#    
#    f = open('SSE_U','w')
#    f.close()
#    
#    f = open('SSE_P','w')
#    f.close()
#    
#    f = open('SSE_Z','w')
#    f.close()

def print_figures(opt_type):
    
    initialize_files()  
        
    if opt_type == 'scan':

        # For scanning the parameter space
        
        nReruns = int(np.loadtxt('iRerun'))
        eps_sig_reruns = np.loadtxt('eps_Sigma_all',skiprows=1)
        eps_reruns = eps_sig_reruns[:,0]
        sig_reruns = eps_sig_reruns[:,1]
        eps = np.unique(eps_reruns)
        sig = np.unique(sig_reruns)
        neps = len(eps)
        nsig = len(sig) 
        
        SSErhoL = np.empty([neps,nsig])
        SSEPsat = np.empty([neps,nsig])
        SSErhov = np.empty([neps,nsig])
        SSEU = np.empty([neps,nsig])
        SSEP = np.empty([neps,nsig])
        SSEZ = np.empty([neps,nsig])
    
        iRerun = 1
        for ieps in range(neps):
            for isig in range(nsig):
                SSErhoL[ieps,isig], SSEPsat[ieps,isig], SSErhov[ieps,isig], SSEU[ieps,isig], SSEP[ieps,isig], SSEZ[ieps,isig] = analyze_ITIC(iRerun)
                iRerun += 1
                
        SSErhoL = np.log10(SSErhoL)
        SSEPsat = np.log10(SSEPsat)
        SSErhov = np.log10(SSErhov)
        SSEU = np.log10(SSEU)
        SSEP = np.log10(SSEP)
        SSEZ = np.log10(SSEZ)
                
        f = plt.figure()
        plt.contour(sig,eps,SSErhoL)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE $\rho_l$')
        f.savefig(compound+'_SSErhoL.pdf')
        
        f = plt.figure()
        plt.contour(sig,eps,SSEPsat)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE $P_{sat}$')
        f.savefig(compound+'_SSEPsat.pdf')
        
        f = plt.figure()
        plt.contour(sig,eps,SSErhov)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE $\rho_v$')
        f.savefig(compound+'_SSErhov.pdf')
        
        f = plt.figure()
        plt.contour(sig,eps,SSEU)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE U')
        f.savefig(compound+'_SSEU.pdf')
        
        f = plt.figure()
        plt.contour(sig,eps,SSEP)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE P')
        f.savefig(compound+'_SSEP.pdf')
        
        f = plt.figure()
        plt.contour(sig,eps,SSEZ)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE Z')
        f.savefig(compound+'_SSEZ.pdf')
    
    else:
        
        # For iterative
        
        nReruns = int(np.loadtxt('iRerun'))+1
        eps_sig_reruns = np.loadtxt('eps_Sigma_all',skiprows=0)
        eps_reruns = eps_sig_reruns[:,0]
        sig_reruns = eps_sig_reruns[:,1]
        eps = np.unique(eps_reruns)
        sig = np.unique(sig_reruns)
        
        SSErhoL = np.empty(nReruns)
        SSEPsat = np.empty(nReruns)
        SSErhov = np.empty(nReruns)
        SSEU = np.empty(nReruns)
        SSEP = np.empty(nReruns)
        SSEZ = np.empty(nReruns)
        
        for iRerun in range(nReruns):
            SSErhoL[iRerun], SSEPsat[iRerun], SSErhov[iRerun], SSEU[iRerun], SSEP[iRerun], SSEZ[iRerun] = analyze_ITIC(iRerun)
                
           
        f = plt.figure()
        plt.semilogy(SSErhoL,marker='o',linestyle='none')
        plt.xlabel('Iteration')
        plt.ylabel('SSE')
        f.savefig(compound+'_SSErhoL.pdf')
        
        f = plt.figure()
        plt.semilogy(SSEPsat,marker='o',linestyle='none')
        plt.xlabel('Iteration')
        plt.ylabel('SSE')
        f.savefig(compound+'_SSEPsat.pdf')
        
        f = plt.figure()
        plt.semilogy(SSErhov,marker='o',linestyle='none')
        plt.xlabel('Iteration')
        plt.ylabel('SSE')
        f.savefig(compound+'_SSErhov.pdf')
        
        f = plt.figure()
        plt.semilogy(SSEU,marker='o',linestyle='none')
        plt.xlabel('Iteration')
        plt.ylabel('SSE')
        f.savefig(compound+'_SSEU.pdf')
        
        f = plt.figure()
        plt.semilogy(SSEP,marker='o',linestyle='none')
        plt.xlabel('Iteration')
        plt.ylabel('SSE')
        f.savefig(compound+'_SSEP.pdf')
        
        f = plt.figure()
        plt.semilogy(SSEZ,marker='o',linestyle='none')
        plt.xlabel('Iteration')
        plt.ylabel('SSE')
        f.savefig(compound+'_SSEZ.pdf')
        
        SSErhoL = np.log10(SSErhoL)
        SSEPsat = np.log10(SSEPsat)
        SSErhov = np.log10(SSErhov)
        SSEU = np.log10(SSEU)
        SSEP = np.log10(SSEP)
        SSEZ = np.log10(SSEZ)
        
        f = plt.figure()
        plt.scatter(sig_reruns[1:],eps_reruns[1:],c=SSErhoL[1:],cmap='Blues',label='Iterations')
        plt.scatter(sig_reruns[0],eps_reruns[0],marker='x',c=SSErhoL[0],cmap='Blues',label='Reference')
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.legend()
        plt.title(r'$\rho_l$')
        ax = plt.colorbar()
        ax.set_label('log(SSE)')
        f.savefig(compound+'_eps_sig_rhoL.pdf')
        
        f = plt.figure()
        plt.scatter(sig_reruns,eps_reruns,c=SSEPsat)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE $P_{sat}$')
        f.savefig(compound+'_eps_sig_SSEPsat.pdf')
        
        f = plt.figure()
        plt.scatter(sig_reruns,eps_reruns,c=SSErhov)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE $\rho_v$')
        f.savefig(compound+'_eps_sig_SSErhov.pdf')
        
        f = plt.figure()
        plt.scatter(sig_reruns,eps_reruns,c=SSEU)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE U')
        f.savefig(compound+'_eps_sig_SSEU.pdf')
        
        f = plt.figure()
        plt.scatter(sig_reruns,eps_reruns,c=SSEP)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE P')
        f.savefig(compound+'_eps_sig_SSEP.pdf')
        
        f = plt.figure()
        plt.scatter(sig_reruns,eps_reruns,c=SSEZ)
        plt.ylabel('$\epsilon$ (kJ/mol)')
        plt.xlabel('$\sigma$ (nm)')
        plt.ylim([min(eps),max(eps)])
        plt.xlim([min(sig),max(sig)])
        plt.title(r'SSE Z')
        f.savefig(compound+'_eps_sig_SSEZ.pdf')
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-opt","--optimizer",type=str,choices=['fsolve','steep','LBFGSB','leapfrog','scan','points'],help="choose which type of optimizer to use")
    args = parser.parse_args()
    if args.optimizer:
        print_figures(args.optimizer)
    else:
        print('Please specify an optimizer type')

if __name__ == '__main__':
    
    main()