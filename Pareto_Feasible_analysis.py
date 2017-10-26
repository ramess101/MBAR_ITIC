# -*- coding: utf-8 -*-
"""
This code is designed to perform a Pareto front and/or Feasible region analysis

@author: ram9
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy import stats
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mpl
import matplotlib.mlab as mlab
import VLE_model_fit

fpathroot = 'MBAR_PCFR/parameter_space_Mie16/'
model_type = 'MBAR_ref8rr_'
model_type = 'Direct_simulation_rr_'
model_type = 'PCFR_sample_refined_'
RMS_all = np.zeros([441,6])

if model_type == 'PCFR_sample_refined_':
    
    RMS_all = np.loadtxt(fpathroot+model_type+'RMS_all',skiprows=5)
    RMS_all[:,3] = RMS_all[:,3]/400.
    RMS_all[:,4] = RMS_all[:,4]/100.     
    
else:

    RMS_all[:,0] = np.loadtxt(fpathroot+model_type+'RMS_rhoL_all')
    RMS_all[:,1] = np.loadtxt(fpathroot+model_type+'RMS_Psat_all')
    RMS_all[:,2] = np.loadtxt(fpathroot+model_type+'RMS_rhov_all')
    RMS_all[:,3] = np.loadtxt(fpathroot+model_type+'RMS_U_all')/400.
    RMS_all[:,4] = np.loadtxt(fpathroot+model_type+'RMS_P_all')/100.
    RMS_all[:,5] = np.loadtxt(fpathroot+model_type+'RMS_Z_all')

if fpathroot == 'MBAR_PCFR/parameter_space_Mie16/':

    eps_sig_all = np.loadtxt(fpathroot+'eps_sig_lam16_highEps_all',skiprows=2)

elif fpathroot == 'MBAR_PCFR/parameter_space_LJ/':

    eps_sig_all = np.loadtxt(fpathroot+'eps_sig_lam12_all',skiprows=2)
    
if model_type == 'PCFR_sample_refined_':
    
    eps_sig_all = np.loadtxt(fpathroot+'eps_sig_lam16_refined',skiprows=5)

eps_all = eps_sig_all[:,0]
sig_all = eps_sig_all[:,1]

eps_plot = np.unique(eps_all)
sig_plot = np.unique(sig_all)

Pareto_optimal = np.ones(441,dtype=bool)

RMS_sorted = RMS_all[np.argsort(RMS_all[:,0])[::-1]]
eps_sorted = eps_all[np.argsort(RMS_all[:,0])[::-1]]
sig_sorted = sig_all[np.argsort(RMS_all[:,0])[::-1]]

for iPareto in range(len(Pareto_optimal)):
    RMS_test = RMS_sorted[iPareto,1:4]
    for iRerun in range(len(RMS_sorted)-iPareto):
        iRerun += iPareto
        RMS_other = RMS_sorted[iRerun,1:4]
        if np.all(RMS_other < RMS_test):
            Pareto_optimal[iPareto] = False
            break
        
eps_Pareto = eps_sorted[Pareto_optimal]
sig_Pareto = sig_sorted[Pareto_optimal]

TypeB = np.zeros([441,6],dtype=bool)
RMS_unc = np.array([5,0.8,0.5,0.3,1,0.2])

for iTypeB in range(len(TypeB)):
    for jTypeB in range(6):
        if RMS_all[iTypeB,jTypeB] < RMS_unc[jTypeB]:
            TypeB[iTypeB,jTypeB] = True
       
plt.plot(sig_Pareto,eps_Pareto,'bx',label='Pareto front')

if fpathroot == 'MBAR_PCFR/parameter_space_Mie16/':

    plt.plot(0.3783,121.25,'ro',label='Potoff')
    plt.xlim([0.365,0.385])
    plt.ylim([108,128])

elif fpathroot == 'MBAR_PCFR/parameter_space_LJ/':
    
    plt.plot(0.375,98.,'ro',label='TraPPE')
    plt.xlim([0.365,0.385])
    plt.ylim([88,108])
    
plt.xlabel(r'$\sigma$ (nm)')
plt.ylabel(r'$\epsilon$ (K)')
plt.legend()
plt.show()

plt.plot(RMS_sorted[Pareto_optimal])
plt.legend([r'$\rho_l$','$P_{sat}$',r'$\rho_v$','U','P','Z'])
plt.ylabel('RMS')
plt.xlabel('Pareto set')
plt.show()

eps_TypeB = [eps_all[TypeB[:,0]],eps_all[TypeB[:,1]],eps_all[TypeB[:,2]],eps_all[TypeB[:,3]],eps_all[TypeB[:,4]],eps_all[TypeB[:,5]]]
sig_TypeB = [sig_all[TypeB[:,0]],sig_all[TypeB[:,1]],sig_all[TypeB[:,2]],sig_all[TypeB[:,3]],sig_all[TypeB[:,4]],sig_all[TypeB[:,5]]]

plt.plot(sig_TypeB[0],eps_TypeB[0],'gv',label=r'Type B, $\rho_l$')
plt.plot(sig_TypeB[1],eps_TypeB[1],'c^',label=r'Type B, $P_{sat}$')
plt.plot(sig_TypeB[2],eps_TypeB[2],'y<',label=r'Type B, $\rho_v$')
plt.plot(sig_TypeB[3],eps_TypeB[3],'m>',label=r'Type B, $U$')
plt.plot(sig_TypeB[4],eps_TypeB[4],'b8',label=r'Type B, $P$')
plt.plot(sig_TypeB[5],eps_TypeB[5],'rs',label=r'Type B, $Z$')

if fpathroot == 'MBAR_PCFR/parameter_space_Mie16/':

    plt.plot(0.3783,121.25,'ko',label='Potoff')
    plt.xlim([0.365,0.385])
    plt.ylim([108,128])

elif fpathroot == 'MBAR_PCFR/parameter_space_LJ/':
    
    plt.plot(0.375,98.,'ko',label='TraPPE')
    plt.xlim([0.365,0.385])
    plt.ylim([88,108])
plt.xlabel(r'$\sigma$ (nm)')
plt.ylabel(r'$\epsilon$ (K)')
plt.legend()
plt.show()

plt.contour(sig_plot,eps_plot,RMS_all[:,0].reshape(21,21))
plt.show()

plt.contour(sig_plot,eps_plot,RMS_all[:,1].reshape(21,21))
plt.show()

plt.contour(sig_plot,eps_plot,RMS_all[:,2].reshape(21,21))
plt.show()

plt.contour(sig_plot,eps_plot,RMS_all[:,3].reshape(21,21))
plt.show()

plt.contour(sig_plot,eps_plot,RMS_all[:,4].reshape(21,21))
plt.show()

plt.contour(sig_plot,eps_plot,RMS_all[:,5].reshape(21,21))
plt.show()
    
#UPZ = np.loadtxt(fpathroot+model_type+str(iRerun)+ending)
#           
#USim = UPZ[:,0]
#PSim = UPZ[:,2]
#ZSim = UPZ[:,4]
#        
#VLE = np.loadtxt(fpathroot+'ITIC_'+model_type+str(iRerun)+ending,skiprows=1)
#
#Tsat, rhoLSim, PsatSim, rhovSim = VLE[:,0], VLE[:,1], VLE[:,2], VLE[:,3]