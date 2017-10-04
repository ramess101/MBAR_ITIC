from __future__ import division
import numpy as np 
import os, sys, argparse, shutil
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import time
import scipy.integrate as integrate
from scipy.optimize import minimize

fpathroot = 'parameter_space_Mie16/'
nReruns = 441
nStates = 19

#Before running script run, "pip install pymbar, pip install CoolProp"

#compound='ETHANE'
compound='Ethane'
#REFPROP_path='/home/ram9/REFPROP-cmake/build/' #Change this for a different system
#
#CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH,REFPROP_path)
#
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

###

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
            
def run_analysis(fpathroot,model_type,ending):
    
    for iRerun in range(nReruns):
        if model_type != 'Constant_rr':
            iRerun += 1
        if model_type == 'MBAR_ref8rr':
            iRerun += 8
        
        UPZ = np.loadtxt(fpathroot+model_type+str(iRerun)+ending)
    
        USim = UPZ[:,0]
        ZSim = UPZ[:,4]
    
        Tsat, rhoLSim, PsatSim, rhovSim = ITIC_calc(USim, ZSim)

        f = open(fpathroot+'ITIC_'+model_type+str(iRerun)+ending,'w')
        f.write('Tsat (K)\trhoL (kg/m3)\tPsat (bar)\trhov (kg/m3)')
        for Tsatprint,rhoLprint,Psatprint,rhovprint in zip(Tsat,rhoLSim,PsatSim,rhovSim):
            f.write('\n'+str(Tsatprint))
            f.write('\t'+str(rhoLprint))
            f.write('\t'+str(Psatprint))
            f.write('\t'+str(rhovprint))
        f.close()

def calc_objective(fpathroot,model_type,ending):    

    for iRerun in range(nReruns):
        if model_type != 'Constant_rr':
            iRerun += 1
        if model_type == 'MBAR_ref8rr':
            iRerun += 8
                    
        UPZ = np.loadtxt(fpathroot+model_type+str(iRerun)+ending)
           
        USim = UPZ[:,0]
        PSim = UPZ[:,2]
        ZSim = UPZ[:,4]
                
        VLE = np.loadtxt(fpathroot+'ITIC_'+model_type+str(iRerun)+ending,skiprows=1)
        
        Tsat, rhoLSim, PsatSim, rhovSim = VLE[:,0], VLE[:,1], VLE[:,2], VLE[:,3]
        
        if np.any(Tsat[Tsat<RP_TC]>RP_Tmin):
        
            RP_rhoL = CP.PropsSI('D','T',Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)],'Q',0,'REFPROP::'+compound) #[kg/m3]   
            RP_rhov = CP.PropsSI('D','T',Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)],'Q',1,'REFPROP::'+compound) #[kg/m3]
            RP_Psat = CP.PropsSI('P','T',Tsat[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)],'Q',1,'REFPROP::'+compound)/100000. #[bar]
        
            devrhoL = rhoLSim[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)] - RP_rhoL #In case Tsat is greater than RP_TC
            devPsat = PsatSim[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)] - RP_Psat
            devrhov = rhovSim[np.logical_and(RP_Tmin<Tsat,Tsat<RP_TC)] - RP_rhov
                              
        else:
            
            devrhoL = np.array([1e8])
            devPsat = np.array([1e8])
            devrhov = np.array([1e8])
                         
        devU = USim - RP_U_depN
        devP = PSim - RP_P
        devZ = ZSim - RP_Z
           
        SSErhoL = np.sum(np.power(devrhoL,2))
        SSEPsat = np.sum(np.power(devPsat,2)) 
        SSErhov = np.sum(np.power(devrhov,2)) 
        SSEU = np.sum(np.power(devU,2))
        SSEP = np.sum(np.power(devP,2))
        SSEZ = np.sum(np.power(devZ,2))
        
        RMSrhoL = np.sqrt(SSErhoL/len(devrhoL))
        RMSPsat = np.sqrt(SSEPsat/len(devPsat))
        RMSrhov = np.sqrt(SSErhov/len(devrhov))
        RMSU = np.sqrt(SSEU/len(devU))
        RMSP = np.sqrt(SSEP/len(devP))
        RMSZ = np.sqrt(SSEZ/len(devZ))
        
#        f = open(fpathroot+'SSE_rhoL_all','a')
#        f.write('\n'+str(SSErhoL))
#        f.close()
#        
#        f = open(fpathroot+'SSE_Psat_all','a')
#        f.write('\n'+str(SSEPsat))
#        f.close()
#        
#        f = open(fpathroot+'SSE_rhov_all','a')
#        f.write('\n'+str(SSErhov))
#        f.close()
#        
#        f = open(fpathroot+'SSE_U_all','a')
#        f.write('\n'+str(SSEU))
#        f.close()
#        
#        f = open(fpathroot+'SSE_P_all','a')
#        f.write('\n'+str(SSEP))
#        f.close()
#        
#        f = open(fpathroot+'SSE_Z_all','a')
#        f.write('\n'+str(SSEZ))
#        f.close() 

        f = open(fpathroot+model_type+'_RMS_rhoL_all','a')
        f.write('\n'+str(RMSrhoL))
        f.close()
        
        f = open(fpathroot+model_type+'_RMS_Psat_all','a')
        f.write('\n'+str(RMSPsat))
        f.close()
        
        f = open(fpathroot+model_type+'_RMS_rhov_all','a')
        f.write('\n'+str(RMSrhov))
        f.close()
        
        f = open(fpathroot+model_type+'_RMS_U_all','a')
        f.write('\n'+str(RMSU))
        f.close()
        
        f = open(fpathroot+model_type+'_RMS_P_all','a')
        f.write('\n'+str(RMSP))
        f.close()
        
        f = open(fpathroot+model_type+'_RMS_Z_all','a')
        f.write('\n'+str(RMSZ))
        f.close()     
                
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt","--optimizer",type=str,choices=['fsolve','steep','LBFGSB','leapfrog','scan','points','SLSQP'],help="choose which type of optimizer to use")
    parser.add_argument("-prop","--properties",type=str,nargs='+',choices=['rhoL','Psat','rhov','P','U','Z'],help="choose one or more properties to use in optimization" )
    args = parser.parse_args()
    
    ending = '_lam16_highEps'
    
    for model_type in ['MBAR_ref8rr']: #['Direct_simulation_rr']: #['MBAR_ref8rr']: #['MBAR_ref0rr', 'PCFR_ref0rr','Constant_rr']:
    
        #run_analysis(fpathroot,model_type,ending)
        calc_objective(fpathroot,model_type,ending)

if __name__ == '__main__':
    '''
    python optimization_Mie_ITIC_multiple_refs.py --optimizer XX --properties XX --lam XX
  
    "--iRerun is required to specify which rerun we are currently on"
    '''

    main()
