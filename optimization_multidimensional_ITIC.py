from __future__ import division
import numpy as np 
import os, sys, argparse
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

print(os.getcwd())
time.sleep(2)

eps_low = np.loadtxt('eps_low')
eps_guess = np.loadtxt('eps_guess')
eps_high = np.loadtxt('eps_high')

sig_low = np.loadtxt('sig_low')
sig_guess = np.loadtxt('sig_guess')
sig_high = np.loadtxt('sig_high')

TOL = np.loadtxt('TOL_MBAR') 

def U_to_u(U,T): #Converts internal energy into reduced potential energy in NVT ensemble
    beta = 1./(R_g*T)
    u = beta*(U)
    return u

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

iRerun = 0

def objective_ITIC(eps_sig,prop_type): 
    global iRerun
    
    USim, dUSim, PSim, dPSim, ZSim = MBAR_estimates(eps_sig,iRerun)
    # From e0s0
    #USim = np.array([-951.717582,-1772.47135,-2472.33725,-3182.829818,-3954.981049,-4348.541215,-4718.819719,-5048.007819,-5302.871063,-6305.502455,-5993.535972,-5711.665352,-5475.703544,-5160.608038,-4988.807891,-4646.560864,-4519.582458,-4174.794714,-4071.662753])
    #ZSim = np.array([0.67064977,0.479303687,0.401264805,0.476499285,1.0267111,1.603610151,2.543291165,3.870781635,5.759734141,-0.431893514,3.077724536,-0.458331725,1.893955078,-0.454276082,1.119877239,-0.384795952,0.643115142,-0.275134489,0.378439128])
    # eps = 0.925 sig = 0.377, the Tsat[1] was problematic with old code because of concavity
    #USim = np.array([-816.049438,-1549.934965,-2191.476435,-2851.822775,-3547.998367,-3907.713473,-4235.273897,-4521.887026,-4697.156908,-5702.583036,-5403.108685,-5139.894986,-4918.18643,-4655.74991,-4498.777318,-4184.347089,-4076.720214,-3765.608436,-3665.575747,])
    #ZSim = np.array([0.738163935,0.596420397,0.547142197,0.693496956,1.365347839,1.913485127,2.818171987,4.118326937,6.114410971,0.518005855,3.634704592,0.47347307,2.526576993,0.259872236,1.572262794,0.217710563,1.030553571,0.086080502,0.742335511])
    # From REFPROP
    #USim = RP_U_depN
    #ZSim = RP_Z
    
    Tsat, rhoLSim, PsatSim, rhovSim = ITIC_calc(USim, ZSim)
    
    #print(Tsat)
    #print(rhoLSim)
    #print(PsatSim)
    #print(rhovSim)
    
    f = open('ITIC_'+str(iRerun),'w')
    f.write('Tsat (K)\trhoL (kg/m3)\tPsat (bar)\trhov (kg/m3)')
    for Tsatprint,rhoLprint,Psatprint,rhovprint in zip(Tsat,rhoLSim,PsatSim,rhovSim):
        f.write('\n'+str(Tsatprint))
        f.write('\t'+str(rhoLprint))
        f.write('\t'+str(Psatprint))
        f.write('\t'+str(rhovprint))
    f.close()

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
        
#        p_Z_IC = np.polyfit(beta_IC,Z_IC-min(Z_IC),2) #Uses the REFPROP ZL
#        p_UT_IC = np.polyfit(beta_IC,UT_IC,1)
#        Z_IC_hat = np.poly1d(p_Z_IC)+min(Z_IC)
#        UT_IC_hat = np.poly1d(p_UT_IC)
#        U_IC_hat = lambda beta: UT_IC_hat(beta)/beta                                     
#                                             
#        beta_sat = np.roots(p_Z_IC).max() #We want the positive root
#        
#        #print(U_IC_hat(beta_IT))
#        #print(beta_IT)
#        #print(beta_sat)
#                           
#        Adep_IC[iIC] = integrate.quad(U_IC_hat,beta_IT,beta_sat)[0]
#        Adep_ITIC[iIC] = Adep_IT(rho_IC) + Adep_IC[iIC]
#        
#        #print(Adep_IT(rho_IC))
#        #print(beta_sat)
#        #print(Adep_IC)
#        #print(Adep_ITIC)
#        
#        Z_L = Z_IC_hat(beta_sat) # Should be 0, but really we should iteratively solve for self-consistency
#        #Z_L = min(Z_IC)
#        #print(Z_L)
#        Tsat[iIC] = 1./beta_sat
#        rhoL[iIC] = rho_IC
#            
#        #print(Tsat)
#        #print(rhoL)
#        #print(Psat)
#        
#        B2 = CP.PropsSI('BVIRIAL','T',Tsat[iIC],'Q',1,'REFPROP::'+compound) #[m3/mol]
#        B2 /= Mw #[m3/kg]
#        B3 = CP.PropsSI('CVIRIAL','T',Tsat[iIC],'Q',1,'REFPROP::'+compound) #[m3/mol]2
#        B3 /= Mw**2 #[m3/kg]
#        eq2_14 = lambda(rhov): Adep_ITIC[iIC] + Z_L - 1 + np.log(rhoL[iIC]/rhov) - 2*B2*rhov + 1.5*B3*rhov**2
#        eq2_15 = lambda(rhov): rhov - rhoL[iIC]*np.exp(Adep_ITIC[iIC] + Z_L - 1 - 2*B2*rhov - 1.5*B3*rhov**2)               
#        SE = lambda rhov: (eq2_15(rhov) - 0.)**2
#        guess = (0.1,)
#        rho_c_RP = CP.PropsSI('RHOCRIT','REFPROP::'+compound)
#        bnds = ((0., rho_c_RP),)
#        opt = minimize(SE,guess,bounds=bnds)
#        rhov[iIC] = opt.x[0] #[kg/m3]
#        
#        Zv = (1. + B2*rhov[iIC] + B3*rhov[iIC]**2)
#        Psat[iIC] = Zv * rhov[iIC] * R_g * Tsat[iIC] / Mw #[kPa]
#        Psat[iIC] /= 100. #[bar]
        
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

#print(objective_ITIC(1.))
          
iEpsRef = int(np.loadtxt('../iEpsref'))
iSigmaRef = int(np.loadtxt('../iSigref'))

def MBAR_estimates(eps_sig,iRerun):
    
    #eps = eps.tolist()

    f = open('eps_it','w')
    f.write(str(eps_sig[0]))
    f.close()
    
    f = open('eps_all','a')
    f.write('\n'+str(eps_sig[0]))
    f.close()
    
    f = open('sig_it','w')
    f.write(str(eps_sig[1]))
    f.close()
    
    f = open('sig_all','a')
    f.write('\n'+str(eps_sig[1]))
    f.close()
    
    f = open('iRerun','w')
    f.write(str(iRerun))
    f.close()
    
    subprocess.call("./EthaneRerunITIC_subprocess")

    g_start = 28 #Row where data starts in g_energy output
    g_t = 0 #Column for the snapshot time
    g_en = 2 #Column where the potential energy is located
    g_T = 4 #Column where T is located
    g_p = 5 #Column where p is located

    iSets = [0, int(iRerun)]
    
    nSets = len(iSets)
    
    N_k = [0]*nSets #This makes a list of nSets elements, need to be 0 not 0. for MBAR to work
        
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
    Z_rerun = Z_MBAR[:,1]
    
    #print(U_rerun)
    #print(dU_rerun)
    #print(P_rerun)
    #print(dP_rerun)
    
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
    
    return U_rerun, dU_rerun, P_rerun, dP_rerun, Z_rerun

def GOLDEN(AX,BX,CX,TOL):

    R_ratio=0.61803399
    C_ratio=1.-R_ratio
    
    X0 = AX
    X3 = CX
    #iRerun=0
    if np.abs(CX-BX) > np.abs(BX-AX):
        X1 = BX
        X2 = BX + C_ratio*(CX-BX)
        F1 = objective_ITIC(X1)
        #iRerun += 1
        F2 = objective_ITIC(X2)
        #f = open('F_all','a')
        #f.write('\n'+str(F1))
        #f.write('\n'+str(F2))
        #f.close()
    else:
        X2 = BX
        X1 = BX - C_ratio*(BX-AX)
        F2 = objective_ITIC(X2)
        #iRerun += 1
        F1 = objective_ITIC(X1)
        #f = open('F_all','a')
        #f.write('\n'+str(F2))
        #f.write('\n'+str(F1))
        #f.close()

    #print(X0,X1,X2,X3)
    while np.abs(X3-X0) > TOL*(np.abs(X1)+np.abs(X2)):
    #for i in np.arange(0,50):
        if F2 < F1:
            X0 = X1
            X1 = X2
            X2 = R_ratio*X1 + C_ratio*X3
            F1 = F2
            eps_it = X2
            F2 = objective_ITIC(X2)
            F_it = F2
            #print(X0,X1,X2,X3)
        else:
            X3 = X2
            X2 = X1
            X1 = R_ratio*X2 + C_ratio*X0
            F2 = F1
            eps_it = X1
            F1 = objective_ITIC(X1)
            F_it = F1
            #print(X0,X1,X2,X3)
        #iRerun += 1
        
        #f = open('F_ITIC_all','a')
        #f.write('\n'+str(F_it))
        #f.close()
        
    if F1 < F2:
        GOLDEN = F1
        XMIN = X1
    else:
        GOLDEN = F2
        XMIN = X2
    
    return XMIN, GOLDEN  

# Updates after each step
def deriv_xj(fun,f0,x,dx,j,equation='5.93'):
    dx_j = 0*dx
    dx_j[j] = dx[j]
    if equation == '5.93':
        f_plus1 = fun(x+dx_j)
        deriv = (f_plus1 - f0)/dx[j]
    elif equation == '5.96':
        f_plus1 = fun(x+dx_j)
        f_minus1 = fun(x-dx_j)
        deriv = (f_plus1 - f_minus1)/2./dx[j]
    elif equation == '5.106':
        f = np.empty([4])
        f[0] = fun(x-2.*dx_j)
        f[1] = fun(x-dx_j)
        f[2] = fun(x+dx_j)
        f[3] = fun(x+2.*dx_j)
        deriv = (f[0] - 8.*f[1] + 8.*f[2] - f[3])/12./dx[j]
    return deriv

def steep_descent(fun,x_guess,bounds,dx,tol,tx,max_it=20,max_fun=30):
    global iRerun
    conv = False
    x_n = x_guess
    deltax = np.zeros(len(x_n))
    it = 0
    f0 = fun(x_n) #Since this method needs f0 anyways, might be better to use Equation 5.93 instead of 5.96
    while not conv and it < max_it and iRerun < max_fun:
        for j, xj in enumerate(x_n):
            conv_tx = False
            while not conv_tx:
                deltax[j] = tx[j]*deriv_xj(fun,f0,x_n,dx,j)
                x_n[j] = xj - deltax[j] 
        #            x_n[j] = min(np.array([max(np.array([bounds[j][0],x_n[j]])),bounds[j][1]]))
        #            deltax[j] = xj - x_n[j]
                if x_n[j] < bounds[j][0]:
                    x_n[j] = bounds[j][0]
                    deltax[j] = xj - x_n[j]
                elif x_n[j] > bounds[j][1]:
                    x_n[j] = bounds[j][1]
                    deltax[j] = xj - x_n[j]
                f_step = fun(x_n)
                if f_step < f0:
                    conv_tx = True
                    f0 = f_step
                else:
                    tx[j] /= 2.
        it += 1
        if (np.abs(deltax) - np.abs(tol) < 0).all():
            conv = True
        
    return x_n, it

def leapfrog(fun,x_guess,bounds,tol,max_it=100):
    dim = len(x_guess)
    nplayers = 10*dim
    players = np.random.random([nplayers,dim])
    for idim in range(dim):
        players[:,idim] *= (bounds[idim][1]-bounds[idim][0])
        players[:,idim] += bounds[idim][0]
    fplayers = np.empty(nplayers)
    for iplayers in range(nplayers):
        fplayers[iplayers] = fun(players[iplayers,:])
    #print(fplayers)
    conv = False
    it = 0
    bnds_trial = np.empty([dim,2])
    while not conv and it < max_it:
        best = fplayers.min()
        worst = fplayers.max()
        ibest = fplayers.argmin()
        iworst = fplayers.argmax()
        dx = players[ibest,:] - players[iworst,:]
        best_player = players[ibest,:]
        extreme_trial = best_player + dx
        #print(best,worst,ibest,iworst)
        #print(dx)
        #print(extreme_trial)
        for idim in range(dim):
            extreme_trial[idim] = min(np.array([max(np.array([bounds[idim][0],extreme_trial[idim]])),bounds[idim][1]]))
            bnds_trial[idim][0] = min([extreme_trial[idim],best_player[idim]])
            bnds_trial[idim][1] = max([extreme_trial[idim],best_player[idim]])
        #print(bnds_trial)
        xtrial = np.random.random(dim)
        for idim in range(dim):
            xtrial[idim] *= (bnds_trial[idim][1]-bnds_trial[idim][0])
            xtrial[idim] += bnds_trial[idim][0]             
        fplayers[iworst] = fun(xtrial)
        players[iworst,:] = xtrial
        it += 1
        #plt.scatter(players[:,0],players[:,1])
        #plt.scatter(xtrial[0],xtrial[1])
        #plt.show()
        if (np.abs(dx) - np.abs(tol) < 0).all():
            conv = True
    ibest = fplayers.argmin()
    best_player = players[ibest,:]
    #print(players[ibest,:])
    return best_player

def call_optimizers(opt_type,prop_type):
    
    objective = lambda eps_sig: objective_ITIC(eps_sig,prop_type)

    eps_opt = 0.
    sig_opt = 0.    
    
    eps_sig_guess = np.array([eps_guess,sig_guess])
    #print(eps_sig_guess)
    
    d_eps_sig = np.array([0.0001,0.0001])
    tol_eps_sig = np.array([0.0001,0.0001])
    t_eps_sig = d_eps_sig
         
    bnds = ((eps_low,eps_high),(sig_low,sig_high)) 
       
    #print(bnds)
    
    if opt_type == 'fsolve':
    
        eps_opt, sig_opt = fsolve(objective,eps_sig_guess,epsfcn=1e-4,xtol=1e-4) #This resulted in some 'nan'
    
    elif opt_type == 'steep':
                                                    
        eps_sig_opt, iterations = steep_descent(objective,eps_sig_guess,bnds,d_eps_sig,tol_eps_sig,t_eps_sig)
        
        eps_opt = eps_sig_opt[0]
        sig_opt = eps_sig_opt[1]

    elif opt_type =='LBFGSB':
         
        sol = minimize(objective,eps_sig_guess,method='L-BFGS-B',bounds=bnds,options={'eps':1e-5,'maxiter':50,'maxfun':100}) #'eps' accounts for the algorithm wanting to take too small of a step change for the Jacobian that Gromacs does not distinguish between the different force fields
        eps_opt = sol.x[0]
        sig_opt = sol.x[1]
    
    elif opt_type == 'leapfrog': 
        
        # For leapfrog algorithm
        objective(eps_sig_guess) #To call objective before running loop
        
        eps_sig_opt = leapfrog(objective,eps_sig_guess,bnds,tol_eps_sig)
        eps_opt = eps_sig_opt[0]
        sig_opt = eps_sig_opt[1]
    
    elif opt_type == 'scan':
        # For scanning the parameter space
        
        objective(eps_sig_guess) #To call objective before running loop
        
        for iEps, eps_sim in enumerate(np.linspace(eps_low,eps_high,40)):
            for iSig, sig_sim in enumerate(np.linspace(sig_low,sig_high,40)):
                eps_sig_sim = np.array([eps_sim,sig_sim])
                objective(eps_sig_sim)
        
    elif opt_type == 'points':
        objective(eps_sig_guess)
        eps_sig_spec = np.array([0.950877,0.377134])
        objective_ITIC(eps_sig_spec)
    
    return eps_opt, sig_opt        
            
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt","--optimizer",type=str,choices=['fsolve','steep','LBFGSB','leapfrog','scan','points'],help="choose which type of optimizer to use")
    parser.add_argument("-prop","--properties",type=str,nargs='+',choices=['rhoL','Psat','rhov','P','U','Z'],help="choose one or more properties to use in optimization" )
    args = parser.parse_args()
    if args.optimizer:
        eps_opt, sig_opt = call_optimizers(args.optimizer,args.properties)
    else:
        print('Please specify an optimizer type')
        eps_opt = 0.
        sig_opt = 0.
    
#    script = sys.argv[0]
#    opt_type = sys.argv[1]
#    
#    eps_opt, sig_opt = call_optimizers(opt_type)

    if eps_opt == 0. or sig_opt == 0.:
        
        conv_eps = 1

    else:               
    
        f = open('eps_optimal','w')
        f.write(str(eps_opt))
        f.close()
        
        f = open('sig_optimal','w')
        f.write(str(sig_opt))
        f.close()
        
        conv_eps = 0
        
        if iEpsRef > 0:
            eps_opt_previous = np.loadtxt('../e'+str(iEpsRef-1)+'s'+str(iSigmaRef-1)+'/eps_optimal')
            eps_opt_current = eps_opt
            TOL_eps = np.loadtxt('TOL_eps')
            sig_opt_previous = np.loadtxt('../e'+str(iEpsRef-1)+'s'+str(iSigmaRef-1)+'/sig_optimal')
            sig_opt_current = sig_opt
            TOL_sig = np.loadtxt('TOL_sig')
            if np.abs(eps_opt_previous - eps_opt_current) < TOL_eps and np.abs(sig_opt_previous - sig_opt_current) < TOL_sig:
                conv_eps = 1
        
    f = open('conv_eps','w')
    f.write(str(conv_eps))
    f.close()

if __name__ == '__main__':
    
    main()
