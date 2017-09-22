from __future__ import division
import numpy as np 
import os, sys, argparse, shutil
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import CoolProp.CoolProp as CP
#from REFPROP_values import *
import time

#Before running script run, "pip install pymbar, pip install CoolProp"

#compound='ETHANE'
##compound='Ethane'
#REFPROP_path='/home/ram9/REFPROP-cmake/build/' #Change this for a different system
#
#CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH,REFPROP_path)
#
#Mw = CP.PropsSI('M','REFPROP::'+compound) #[kg/mol]
#RP_TC = CP.PropsSI('TCRIT','REFPROP::'+compound)
#RP_Tmin =  CP.PropsSI('TMIN','REFPROP::'+compound)
Mw = 30.069/1000. #[kg/mol]
RP_TC = 305.32 # [K]
RP_Tmin = 90.368 # [K]

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

###

#def REFPROP_UP(TSim,rho_mass,NmolSim,compound):
#    RP_U = CP.PropsSI('UMOLAR','T',TSim,'D',rho_mass,'REFPROP::'+compound) / 1e3 #[kJ/mol]
#    RP_U_ig = CP.PropsSI('UMOLAR','T',TSim,'D',0,'REFPROP::'+compound) / 1e3 #[kJ/mol]
#    RP_U_dep = RP_U - RP_U_ig
#    RP_U_depRT = RP_U_dep / TSim / R_g
#    RP_U_depN = RP_U_dep * NmolSim
#    RP_Z = CP.PropsSI('Z','T',TSim,'D',rho_mass,'REFPROP::'+compound)
#    RP_P = CP.PropsSI('P','T',TSim,'D',rho_mass,'REFPROP::'+compound) / 1e5 #[bar]
#    RP_Z1rho = (RP_Z - 1.)/rho_mass
#
#    f = open('REFPROP_UPZ','w')
#
#    for iState, Temp in enumerate(TSim):
#
#        f.write(str(RP_U_depN[iState])+'\t')
#        f.write(str(RP_P[iState])+'\t')
#        f.write(str(RP_Z[iState])+'\t')
#        f.write(str(RP_Z1rho[iState])+'\n')
#
#    f.close()        
#
#    return RP_U_depN, RP_P, RP_Z, RP_Z1rho
#
##Generate REFPROP values, prints out into a file in the correct directory
#
#RP_U_depN, RP_P, RP_Z, RP_Z1rho = REFPROP_UP(Temp_sim,rho_mass,Nmol_sim,compound)              

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
    
    f = open('ITIC_'+str(iRerun),'w')
    f.write('Tsat (K)\trhoL (kg/m3)\tPsat (bar)\trhov (kg/m3)')
    for Tsatprint,rhoLprint,Psatprint,rhovprint in zip(Tsat,rhoLSim,PsatSim,rhovSim):
        f.write('\n'+str(Tsatprint))
        f.write('\t'+str(rhoLprint))
        f.write('\t'+str(Psatprint))
        f.write('\t'+str(rhovprint))
    f.close()

    RP_rhoL = CP.PropsSI('D','T',Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)],'Q',0,'REFPROP::'+compound) #[kg/m3]   
    RP_rhov = CP.PropsSI('D','T',Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)],'Q',1,'REFPROP::'+compound) #[kg/m3]
    RP_Psat = CP.PropsSI('P','T',Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)],'Q',1,'REFPROP::'+compound)/100000. #[bar]

    devrhoL = rhoLSim[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)] - RP_rhoL #In case Tsat is greater than RP_TC
    devPsat = PsatSim[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)] - RP_Psat
    devrhov = rhovSim[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)] - RP_rhov
                     
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

def compile_simulation_results(fpathroot,iRerun):
                       
    g_start = 28 #Row where data starts in g_energy output
    g_t = 0 #Column for the snapshot time
    g_LJsr = 1 #Column where the 'Lennard-Jones' short-range interactions are located
    g_LJdc = 2 #Column where the 'Lennard-Jones' dispersion corrections are located
    g_en = 3 #Column where the potential energy is located
    g_T = 4 #Column where T is located
    g_p = 5 #Column where p is located
    
    U_avg = np.zeros(nStates)
    dU = np.zeros(nStates)
    P_avg = np.zeros(nStates)
    dP = np.zeros(nStates)
    Z_avg = np.zeros(nStates)
    Z1rho_avg = np.zeros(nStates)
        
    for iState in range(nStates):
        
        rho_state = rho_sim[iState]
        Temp_state = Temp_sim[iState]
        Nstate = Nmol_sim[iState]
        fpath = fpath_all[iState]
                                                    
        en_p = open(fpathroot+fpath+'nvt_prod.xvg','r').readlines()[g_start:] #Read all lines starting at g_start for "state" k

        nSnaps = len(en_p)
        
        time = np.zeros(nSnaps)
        LJsr = np.zeros(nSnaps)
        LJdc = np.zeros(nSnaps)
        ener = np.zeros(nSnaps)
        press = np.zeros(nSnaps)
        Temp = np.zeros(nSnaps)
        
        for frame in xrange(nSnaps):
                
            time[frame] = float(en_p[frame].split()[g_t])
            LJsr[frame] = float(en_p[frame].split()[g_LJsr])
            LJdc[frame] = float(en_p[frame].split()[g_LJdc])
            ener[frame] = float(en_p[frame].split()[g_en])
            press[frame] = float(en_p[frame].split()[g_p])
            Temp[frame] = float(en_p[frame].split()[g_T])
        
        U_total = ener
        LJ_total = LJsr + LJdc
        
        U_avg[iState] = np.mean(U_total)
        dU[iState] = np.std(U_total)/np.sqrt(nSnaps)
        P_avg[iState] = np.mean(press)
        dP[iState] = np.std(press)/np.sqrt(nSnaps)
        Z_avg[iState] = P_avg[iState]/rho_state/Temp_state/R_g * bar_nm3_to_kJ_per_mole #EP [bar] rho_sim [1/nm3] Temp_sim [K] R_g [kJ/mol/K] #There is probably a better way to assign Z_MBAR
        Z1rho_avg[iState] = (Z_avg[iState] - 1.)/rho_state * 1000. #[ml/gm]

    f = open('Direct_simulation_rr'+str(iRerun),'w')
        
    for iState in range(nStates):
            
        f.write(str(U_avg[iState])+'\t')
        f.write(str(dU[iState])+'\t')
        f.write(str(P_avg[iState])+'\t')
        f.write(str(dP[iState])+'\t')
        f.write(str(Z_avg[iState])+'\t')
        f.write(str(Z1rho_avg[iState])+'\t')
        f.write(str(nSnaps)+'\n')
    
    f.close()
    
    return U_avg, dU, P_avg, dP, Z_avg

def loop_dir():
    eps_range = list(range(88,109))
    sig_range = np.linspace(0.365,0.385,21)

    iRerun = 1
    
    for eps_sample in eps_range:
        for sig_sample in sig_range:
            
            fpathroot = str(eps_sample)+'_%.3f/' % sig_sample
            print(fpathroot)
            
            run_analysis(fpathroot,iRerun)
            
            iRerun += 1
            
def run_analysis(fpathroot,iRerun):
    
    USim, dU, PSim, dP, ZSim = compile_simulation_results(fpathroot,iRerun)
    
    Tsat, rhoLSim, PsatSim, rhovSim = ITIC_calc(USim, ZSim)
    
    #print(Tsat)
    #print(rhoLSim)
    #print(PsatSim)
    #print(rhovSim)
    
    f = open('Direct_simulation_ITIC_'+str(iRerun),'w')
    f.write('Tsat (K)\trhoL (kg/m3)\tPsat (bar)\trhov (kg/m3)')
    for Tsatprint,rhoLprint,Psatprint,rhovprint in zip(Tsat,rhoLSim,PsatSim,rhovSim):
        f.write('\n'+str(Tsatprint))
        f.write('\t'+str(rhoLprint))
        f.write('\t'+str(Psatprint))
        f.write('\t'+str(rhovprint))
    f.close()             
            
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt","--optimizer",type=str,choices=['fsolve','steep','LBFGSB','leapfrog','scan','points','SLSQP'],help="choose which type of optimizer to use")
    parser.add_argument("-prop","--properties",type=str,nargs='+',choices=['rhoL','Psat','rhov','P','U','Z'],help="choose one or more properties to use in optimization" )
    args = parser.parse_args()
    
    loop_dir()

if __name__ == '__main__':
    '''
    python optimization_Mie_ITIC_multiple_refs.py --optimizer XX --properties XX --lam XX
  
    "--iRerun is required to specify which rerun we are currently on"
    '''

    main()
