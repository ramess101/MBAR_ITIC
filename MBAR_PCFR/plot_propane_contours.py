# -*- coding: utf-8 -*-
"""
Plots the contours for propane

@author: ram9
"""

import numpy as np
import matplotlib.pyplot as plt

eps_range = np.linspace(50,70,21)
sig_range = np.linspace(0.385,0.415,31)

eps_range_PCFR = np.linspace(50,70,11)
sig_range_PCFR = np.linspace(0.385,0.415,11)

eps_range_MBAR = np.linspace(50,70,51)
sig_range_MBAR = np.linspace(0.385,0.415,51)

eps_range_MBAR_coarse = np.linspace(50,70,21)
sig_range_MBAR_coarse = np.linspace(0.385,0.415,21)

eps_ref = np.array([54]*8)
sig_ref = np.array([0.395,	0.3973,	0.3996,	0.4019, 0.3904,	0.3927,	0.4043,	0.4066])

sig_Potoff = 0.399
eps_Potoff = 61.0

path_root = 'propane_parameter_space_Mie16/'

# For ethane
#eps_range_PCFR = np.linspace(108,128,11)
#sig_range_PCFR = np.linspace(0.365,0.385,11)
#
#sig_Potoff = 0.3783
#eps_Potoff = 121.25
#
#path_root = 'C2H6_parameter_space_Mie16/'

RMS_MBAR = np.loadtxt(path_root+'PCFR_refs_Psat/Tsat_ITIC/RMS_all',skiprows=1)

RMS_rhol_direct = np.loadtxt(path_root+'Direct_simulation_rr_RMS_rhoL_all')
RMS_rhol_MBAR = RMS_MBAR[:,0]
#RMS_rhol_PCFR = np.loadtxt(path_root+'PCFR_ref0rr_RMS_rhoL_all')

RMS_MBAR_butane = np.loadtxt('H:\Basis_Functions_Butane\RMS_all',skiprows=1)

RMS_rhol_MBAR_butane = RMS_MBAR_butane[:,0]
RMS_Psat_MBAR_butane = RMS_MBAR_butane[:,1]

#RMS_MBAR_octane = np.loadtxt('H:/Basis_Functions_Octane/ref7_fixed/Refined/RMS_all',skiprows=1)
#RMS_MBAR_octane = np.loadtxt('H:/Basis_Functions_Octane/ref7_fixed/Without_ref0/RMS_all',skiprows=1)
RMS_MBAR_octane = np.loadtxt('H:/Basis_Functions_Octane/ref7_fixed/Without_ref0/Refined/RMS_all',skiprows=1)
RMS_rhol_MBAR_octane = RMS_MBAR_octane[:,0]
RMS_logPsat_MBAR_octane = RMS_MBAR_octane[:,1]

# For now I am manually fixing the contours since the 0.395 reference is having problems
#sig_cut_low = 0.393
#sig_cut_high = 0.397
#
#sig_below_cutoff = sig_range_MBAR[sig_range_MBAR<sig_cut_low]
#sig_above_cutoff = sig_range_MBAR[sig_range_MBAR>sig_cut_high]
#nsig_fixed = len(sig_below_cutoff) + len(sig_above_cutoff)
#
#sig_fixed = np.zeros(nsig_fixed)
#
#RMS_rhol_MBAR_octane_fixed = np.zeros([nsig_fixed,len(eps_range_MBAR)])
#
#sig_fixed = []
#RMS_rhol_MBAR_octane_fixed = []
#RMS_logPsat_MBAR_octane_fixed = []
#
#counter = 0
#for iEps, eps_sim in enumerate(eps_range_MBAR):
#    for iSig, sig_sim in enumerate(sig_range_MBAR):
#        if sig_sim < sig_cut_low or sig_sim > sig_cut_high:
#            sig_fixed.append(sig_sim)
#            RMS_rhol_MBAR_octane_fixed.append(RMS_rhol_MBAR_octane[counter])
#            RMS_logPsat_MBAR_octane_fixed.append(RMS_logPsat_MBAR_octane[counter])
#        counter+=1
#
#sig_fixed = np.unique(sig_fixed)
#RMS_rhol_MBAR_octane_fixed = np.array(RMS_rhol_MBAR_octane_fixed)
#RMS_logPsat_MBAR_octane_fixed = np.array(RMS_logPsat_MBAR_octane_fixed)

contour_lines = [2,4,6,8,10,12,14,16]

f, axarr = plt.subplots(nrows=2,ncols=1,figsize=(10,12))   
    
plt.tight_layout(pad=3.4,rect=[-0.02,-0.01,1.04,1.02])
    
plt.text(0.3825,93,'a)',fontsize=24) 
plt.text(0.3825,69,'b)',fontsize=24)  

CS1 = axarr[0].contour(sig_range,eps_range,RMS_rhol_direct.reshape([len(eps_range),len(sig_range)]),contour_lines,colors='r',linestyles='solid')
CS2 = axarr[0].contour(sig_range_MBAR,eps_range_MBAR,RMS_rhol_MBAR.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='b',linestyles='dotted')
CS3 = axarr[0].contour(sig_range_MBAR,eps_range_MBAR,RMS_rhol_MBAR_butane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines[1:2],colors='g',linestyles='dashed')
CS4 = axarr[0].contour(sig_range_MBAR,eps_range_MBAR,RMS_rhol_MBAR_octane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines[1:2],colors='orange',linestyles='dashdot')
#CS4 = axarr[0].contour(sig_fixed,eps_range_MBAR,RMS_rhol_MBAR_octane_fixed.reshape([len(eps_range_MBAR),len(sig_fixed)]),contour_lines[1:2],colors='orange',linestyles='dashdot')
#CS4 = axarr[0].contour(sig_range_MBAR_coarse,eps_range_MBAR_coarse,RMS_rhol_MBAR_octane.reshape([len(eps_range_MBAR_coarse),len(sig_range_MBAR_coarse)]),contour_lines[1:2],colors='orange',linestyles='dashdot')
#CS3 = axarr[0].contour(sig_range_PCFR,eps_range_PCFR,RMS_rhol_PCFR.reshape([len(eps_range_PCFR),len(sig_range_PCFR)]))
axarr[0].clabel(CS1,inline=1,fontsize=16,colors='w',fmt='%1.0f',manual=[(0.395,51),(0.395,52),(0.395,53),(0.395,56),(0.395,57),(0.395,58),(0.395,61),(0.395,62),(0.395,63),(0.395,64),(0.395,65),(0.395,66),(0.395,67),(0.395,68),(0.395,69),(0.400,51)])
axarr[0].clabel(CS2,inline=1,fontsize=16,colors='k',fmt='%1.0f',manual=[(0.395,51),(0.395,52),(0.395,53),(0.395,54),(0.395,56),(0.395,57),(0.395,58),(0.395,61),(0.395,62),(0.395,63),(0.395,64),(0.395,65),(0.395,67),(0.395,68),(0.395,69),(0.400,51)])
#axarr[0].clabel(CS3,inline=1,fontsize=16,colors='g',fmt='%1.0f',manual=[(0.392,60.5),(0.398,60.5),(0.401,60),(0.388,58)]) #(0.396,58),(0.396,60),(0.396,61),(0.396,62)
axarr[0].clabel(CS3,inline=1,fontsize=16,colors='g',fmt='%1.0f',manual=[(0.401,60),(0.388,58)])
axarr[0].clabel(CS4,inline=1,fontsize=16,colors='orange',fmt='%1.0f',manual=[(0.401,62),(0.3965,59)])
axarr[0].plot(sig_Potoff,eps_Potoff,'kx',markersize=10,label=r'Potoff, $\lambda = 16$')
#axarr[0].plot(sig_ref,eps_ref,'bs',mfc='None',markersize=10,label='References (PCFR-optimal)')
axarr[0].plot(sig_ref,eps_ref,'bs',mfc='None',markersize=10,label='References, propane')
axarr[0].plot(sig_ref,eps_ref+2.,'g^',mfc='None',markersize=10,label='References, $n$-butane')
axarr[0].plot(sig_ref,eps_ref+7.,color='orange',linestyle='None',marker='v',mfc='None',markersize=10,label='References, $n$-octane')
axarr[0].set_xlabel(r'$\sigma_{\rm CH_2}$ (nm)',fontsize=26)
axarr[0].set_ylabel(r'$\epsilon_{\rm CH_2}$ (K)',fontsize=26)
axarr[0].set_yticks([50,55,60,65,70])
axarr[0].set_title(r'RMS of $\rho_{\rm l}^{\rm sat}$ (kg/m$^3$)',fontsize=24)# \left(\frac{kg}{m^3}\right)$')

###
#RMS_Psat_direct = np.loadtxt(path_root+'Direct_simulation_rr_RMS_Psat_all')
#RMS_Psat_MBAR = RMS_MBAR[:,1]
##RMS_Psat_PCFR = np.loadtxt(path_root+'PCFR_ref0rr_RMS_Psat_all')
#
#contour_lines = [0.4,0.8,1.2,1.6,2.0]
#
#CS1 = axarr[1].contour(sig_range,eps_range,RMS_Psat_direct.reshape([len(eps_range),len(sig_range)]),contour_lines,colors='r',linestyles='solid')
#CS2 = axarr[1].contour(sig_range_MBAR,eps_range_MBAR,RMS_Psat_MBAR.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='b',linestyles='dotted')
#CS3 = axarr[1].contour(sig_range_MBAR,eps_range_MBAR,RMS_Psat_MBAR_butane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines[0:2],colors='g',linestyles='dashed')
##CS3 = axarr[1].contour(sig_range_PCFR,eps_range_PCFR,RMS_Psat_PCFR.reshape([len(eps_range_PCFR),len(sig_range_PCFR)]))
#axarr[1].clabel(CS1,inline=1,fontsize=16,colors='w',fmt='%1.1f',manual=[(0.396,51),(0.396,54),(0.396,56),(0.396,58),(0.396,60),(0.396,64),(0.396,66),(0.396,68),(0.396,69)])
#axarr[1].clabel(CS2,inline=1,fontsize=16,colors='k',fmt='%1.1f',manual=[(0.396,51),(0.396,54),(0.396,56),(0.396,58),(0.396,60),(0.396,64),(0.396,66),(0.396,68),(0.396,69)])
#axarr[1].clabel(CS3,inline=1,fontsize=16,colors='g',fmt='%1.1f',manual=[(0.394,60),(0.394,62),(0.3935,64),(0.394,66)])
#axarr[1].plot([],[],'r-',label='Direct Simulation, propane')
#axarr[1].plot([],[],'b:',label=r'MBAR, propane')
#axarr[1].plot([],[],'g--',label=r'MBAR, $n$-butane')
##axarr[1].plot(sig_ref,eps_ref,'bs',mfc='None',markersize=10,label='References (PCFR-optimal)')
#axarr[1].plot(sig_ref,eps_ref,'bs',mfc='None',markersize=10,label='References, propane')
#axarr[1].plot(sig_ref,eps_ref+2.,'g^',mfc='None',markersize=10,label='References, $n$-butane')
#axarr[1].plot(sig_Potoff,eps_Potoff,'kx',markersize=10,label=r'Potoff, $\lambda_{\rm CH_2} = 16$')
#axarr[1].set_xlabel(r'$\sigma_{\rm CH_2}$ (nm)')
#axarr[1].set_ylabel(r'$\epsilon_{\rm CH_2}$ (K)')
#axarr[1].set_title(r'RMS of $P_{\rm v}^{\rm sat}$ (bar)')
#axarr[1].set_yticks([50,55,60,65,70])
#axarr[1].legend()

RMS_logPsat_direct = np.loadtxt(path_root+'Direct_simulation_rr_RMS_logPsat_all')
RMS_logPsat_MBAR = np.loadtxt('H:/Basis_Functions_Propane/Mie16/Coarse/ref0_7/MBAR_rr_RMS_logPsat_all')
#RMS_Psat_PCFR = np.loadtxt(path_root+'PCFR_ref0rr_RMS_Psat_all')
RMS_MBAR_butane = np.loadtxt('H:\Basis_Functions_Butane\RMS_all_log',skiprows=1)
RMS_logPsat_MBAR_butane = RMS_MBAR_butane[:,1]

contour_lines=[0.02,0.05,0.10,0.15,0.20]

CS1 = axarr[1].contour(sig_range,eps_range,RMS_logPsat_direct.reshape([len(eps_range),len(sig_range)]),contour_lines,colors='r',linestyles='solid')
CS2 = axarr[1].contour(sig_range_MBAR,eps_range_MBAR,RMS_logPsat_MBAR.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='b',linestyles='dotted')
CS3 = axarr[1].contour(sig_range_MBAR,eps_range_MBAR,RMS_logPsat_MBAR_butane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines[1:2],colors='g',linestyles='dashed')
CS4 = axarr[1].contour(sig_range_MBAR,eps_range_MBAR,RMS_logPsat_MBAR_octane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines[1:2],colors='orange',linestyles='dashdot')
#CS4 = axarr[1].contour(sig_fixed,eps_range_MBAR,RMS_logPsat_MBAR_octane_fixed.reshape([len(eps_range_MBAR),len(sig_fixed)]),contour_lines[1:2],colors='orange',linestyles='dashdot')
#CS4 = axarr[1].contour(sig_range_MBAR_coarse,eps_range_MBAR_coarse,RMS_logPsat_MBAR_octane.reshape([len(eps_range_MBAR_coarse),len(sig_range_MBAR_coarse)]),contour_lines[1:2],colors='orange',linestyles='dashdot')
#CS3 = axarr[1].contour(sig_range_PCFR,eps_range_PCFR,RMS_Psat_PCFR.reshape([len(eps_range_PCFR),len(sig_range_PCFR)]))
axarr[1].clabel(CS1,inline=1,fontsize=16,colors='w',fmt='%1.2f',manual=[(0.396,51),(0.396,54),(0.396,56),(0.396,58),(0.396,60),(0.396,64),(0.396,62),(0.396,66),(0.398,68.5)])
axarr[1].clabel(CS2,inline=1,fontsize=16,colors='k',fmt='%1.2f',manual=[(0.396,51),(0.396,54),(0.396,56),(0.396,58),(0.396,60),(0.396,64),(0.396,62),(0.396,66),(0.398,68.5)])
#axarr[1].clabel(CS3,inline=1,fontsize=16,colors='g',fmt='%1.2f',manual=[(0.392,63),(0.391,64.5),(0.391,62),(0.392,65)])
axarr[1].clabel(CS3,inline=1,fontsize=16,colors='g',fmt='%1.2f',manual=[(0.391,62),(0.392,65)])
axarr[1].clabel(CS4,inline=1,fontsize=16,colors='orange',fmt='%1.2f',manual=[(0.389,63),(0.389,66)])
axarr[1].plot([],[],'r-',label='Direct Simulation, propane')
axarr[1].plot([],[],'b:',label=r'MBAR, propane')
axarr[1].plot([],[],'g--',label=r'MBAR, $n$-butane')
axarr[1].plot([],[],color='orange',linestyle='-.',label=r'MBAR, $n$-octane')
#axarr[1].plot(sig_ref,eps_ref,'bs',mfc='None',markersize=10,label='References (PCFR-optimal)')
axarr[1].plot(sig_ref,eps_ref,'bs',mfc='None',markersize=10,label='References, propane')
axarr[1].plot(sig_ref,eps_ref+2.,'g^',mfc='None',markersize=10,label='References, $n$-butane')
axarr[1].plot(sig_ref,eps_ref+7.,color='orange',linestyle='None',marker='v',mfc='None',markersize=10,label='References, $n$-octane')
axarr[1].plot(sig_Potoff,eps_Potoff,'kx',markersize=10,label=r'Potoff, $\lambda_{\rm CH_2} = 16$')
axarr[1].set_xlabel(r'$\sigma_{\rm CH_2}$ (nm)',fontsize=26)
axarr[1].set_ylabel(r'$\epsilon_{\rm CH_2}$ (K)',fontsize=26)
axarr[1].set_title(r'RMS of log$_{10}\left(P_{\rm v}^{\rm sat}\right)$',fontsize=24)
axarr[1].set_yticks([50,55,60,65,70])
axarr[1].legend(handlelength=1.5,handletextpad=0.01,labelspacing=0.01,borderpad=0.05)

#f.savefig('RMS_rhol_Psat_propane.pdf')
#f.savefig('RMS_rhol_Psat_propane_butane.pdf')
#f.savefig('RMS_rhol_logPsat_propane_butane.pdf')
f.savefig('RMS_rhol_logPsat_propane_butane_octane.pdf')
plt.show()
#
#RMS_Z_direct = np.loadtxt(path_root+'Direct_simulation_rr_RMS_Z_all')
#RMS_Z_MBAR = RMS_MBAR[:,-1]
##RMS_Z_PCFR = np.loadtxt(path_root+'PCFR_ref0rr_RMS_Z_all')
#
##contour_lines = [0.2, 0.4, 0.6,0.8,1.0,1.2,1.4]
#
#plt.contour(sig_range,eps_range,RMS_Z_direct.reshape([len(eps_range),len(sig_range)]),colors='r')
#plt.contour(sig_range_MBAR,eps_range_MBAR,RMS_Z_MBAR.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='b',linestyles='dotted')
##plt.contour(sig_range_PCFR,eps_range_PCFR,RMS_Z_PCFR.reshape([len(eps_range_PCFR),len(sig_range_PCFR)]))
#plt.plot(sig_Potoff,eps_Potoff,'kx',markersize=10,label='Potoff')
#plt.colorbar()
#plt.xlabel(r'$\sigma (nm)$')
#plt.ylabel(r'$\epsilon (K)$')
#plt.title('Z')
#plt.show()
#
#RMS_U_direct = np.loadtxt(path_root+'Direct_simulation_rr_RMS_U_all')
##RMS_U_PCFR = np.loadtxt(path_root+'PCFR_ref0rr_RMS_U_all')/20.
#
#plt.contour(sig_range,eps_range,RMS_U_direct.reshape([len(eps_range),len(sig_range)]))
##plt.contour(sig_range_PCFR,eps_range_PCFR,RMS_U_PCFR.reshape([len(eps_range_PCFR),len(sig_range_PCFR)]))
#plt.plot(sig_Potoff,eps_Potoff,'kx',markersize=10,label='Potoff')
#plt.colorbar()
#plt.xlabel(r'$\sigma (nm)$')
#plt.ylabel(r'$\epsilon (K)$')
#plt.title('$U^{dep}$')
#plt.show()

logp_propane = np.loadtxt(path_root+'PCFR_refs_Psat/logp')
logp_butane = np.loadtxt('H:/Basis_Functions_Butane/logp')
logp_octane = np.loadtxt('H:/Basis_Functions_Octane/ref7_fixed/With_ref0_repeat/Refined/logp')
#logp_octane = np.loadtxt('H:/Basis_Functions_Octane/ref7_fixed/Without_ref0/logp')
logp_transferable = logp_propane + logp_butane + logp_octane

logp_scaled_propane = logp_propane - np.max(logp_propane)
logp_scaled_butane = logp_butane - np.max(logp_butane)
logp_scaled_octane = logp_octane - np.max(logp_octane[logp_octane<1e-30])
logp_scaled_transferable = logp_transferable - np.max(logp_transferable[logp_transferable<1e-30])

prob_propane = np.exp(logp_scaled_propane)
prob_butane = np.exp(logp_scaled_butane)
prob_octane = np.exp(logp_scaled_octane)
prob_transferable = np.exp(logp_scaled_transferable)

f, axarr = plt.subplots(nrows=1,ncols=1,figsize=(10,10))  

contour_lines = [0.1,0.2,0.5,0.9]

CS2 = axarr.contour(sig_range_MBAR,eps_range_MBAR,prob_propane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='b',linestyles='dotted')
CS3 = axarr.contour(sig_range_MBAR,eps_range_MBAR,prob_butane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='g',linestyles='dashed')
CS4 = axarr.contour(sig_range_MBAR,eps_range_MBAR,prob_octane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='orange',linestyles='dashdot')
CS5 = axarr.contour(sig_range_MBAR,eps_range_MBAR,prob_transferable.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='r',linestyles='solid')
#CS4 = axarr.contour(sig_range_MBAR_coarse,eps_range_MBAR_coarse,prob_octane.reshape([len(eps_range_MBAR_coarse),len(sig_range_MBAR_coarse)]),contour_lines,colors='orange',linestyles='dashdot')
axarr.clabel(CS2,inline=0,fontsize=16,colors='b',fmt='%1.1f',manual=[(0.3975,59.5),(0.3975,60),(0.3975,60.5),(0.3975,61),(0.3975,61.5),(0.3975,61.8),(0.3975,62.2),(0.3975,62.5)])
axarr.clabel(CS3,inline=0,fontsize=16,colors='g',fmt='%1.1f',manual=[(0.3965,61),(0.3955,61),(0.3945,61),(0.397,61.5),(0.3985,62),(0.3992,62.2),(0.4,62.5)]) # inline_spacing did not work well
axarr.clabel(CS4,inline=0,fontsize=16,colors='orange',fmt='%1.1f')
axarr.clabel(CS5,inline=0,fontsize=16,colors='r',fmt='%1.1f')
axarr.plot(sig_Potoff,eps_Potoff,'kx',markersize=14,markeredgewidth=5,label=r'Potoff, $\lambda_{\rm CH_2} = 16$')
#axarr.plot(sig_ref,eps_ref,'bs',mfc='None',markersize=10,label='References (PCFR-optimal)')
axarr.plot([],[],'b:',label=r'MBAR, propane')
axarr.plot([],[],'g--',label=r'MBAR, $n$-butane')
axarr.plot([],[],color='orange',linestyle='dashdot',label=r'MBAR, $n$-octane')
axarr.plot([],[],color='r',linestyle='solid',label=r'MBAR, transferable')
axarr.set_xlabel(r'$\sigma_{\rm CH_2}$ (nm)',fontsize=18)
axarr.set_ylabel(r'$\epsilon_{\rm CH_2}$ (K)',fontsize=18)
axarr.set_xlim([0.393,0.402])
axarr.set_ylim([59,63])
axarr.set_yticks([59,60,61,62,63])
axarr.set_xticks([0.393,0.396,0.399,0.402])
axarr.legend(loc=2)
axarr.set_title(r'Normalized Posterior Probability')# \left(\frac{kg}{m^3}\right)$')

f.savefig('prob_propane_butane_octane.pdf')

plt.show()

f, axarr = plt.subplots(nrows=1,ncols=1,figsize=(10,10))  

contour_lines = [-10,-8,-6]

CS2 = axarr.contour(sig_range_MBAR,eps_range_MBAR,logp_propane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='b',linestyles='dotted')
CS3 = axarr.contour(sig_range_MBAR,eps_range_MBAR,logp_butane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='g',linestyles='dashed')
CS4 = axarr.contour(sig_range_MBAR,eps_range_MBAR,logp_octane.reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='orange',linestyles='dashdot')
CS5 = axarr.contour(sig_range_MBAR,eps_range_MBAR,(logp_transferable/3.).reshape([len(eps_range_MBAR),len(sig_range_MBAR)]),contour_lines,colors='r',linestyles='solid')
#CS4 = axarr.contour(sig_range_MBAR_coarse,eps_range_MBAR_coarse,prob_octane.reshape([len(eps_range_MBAR_coarse),len(sig_range_MBAR_coarse)]),contour_lines,colors='orange',linestyles='dashdot')
axarr.clabel(CS2,inline=0,fontsize=16,colors='b',fmt='%1.1f',manual=[(0.3975,59.5),(0.3975,60),(0.3975,60.5),(0.3975,61),(0.3975,61.5),(0.3975,61.8),(0.3975,62.2),(0.3975,62.5)])
axarr.clabel(CS3,inline=0,fontsize=16,colors='g',fmt='%1.1f',manual=[(0.3965,61),(0.3955,61),(0.3945,61),(0.397,61.5),(0.3985,62),(0.3992,62.2),(0.4,62.5)]) # inline_spacing did not work well
axarr.clabel(CS4,inline=0,fontsize=16,colors='orange',fmt='%1.1f')
axarr.clabel(CS5,inline=0,fontsize=16,colors='r',fmt='%1.1f')
axarr.plot(sig_Potoff,eps_Potoff,'kx',markersize=14,markeredgewidth=5,label=r'Potoff, $\lambda_{\rm CH_2} = 16$')
#axarr.plot(sig_ref,eps_ref,'bs',mfc='None',markersize=10,label='References (PCFR-optimal)')
axarr.plot([],[],'b:',label=r'MBAR, propane')
axarr.plot([],[],'g--',label=r'MBAR, $n$-butane')
axarr.plot([],[],color='orange',linestyle='dashdot',label=r'MBAR, $n$-octane')
axarr.plot([],[],color='r',linestyle='solid',label=r'MBAR, transferable')
axarr.set_xlabel(r'$\sigma_{\rm CH_2}$ (nm)',fontsize=18)
axarr.set_ylabel(r'$\epsilon_{\rm CH_2}$ (K)',fontsize=18)
axarr.set_xlim([0.393,0.402])
axarr.set_ylim([59,63])
axarr.set_yticks([59,60,61,62,63])
axarr.set_xticks([0.393,0.396,0.399,0.402])
axarr.legend(loc=2)
axarr.set_title(r'logp')# \left(\frac{kg}{m^3}\right)$')

f.savefig('logp_propane_butane_octane.pdf')

plt.show()

eps_sig_lam_all = np.loadtxt('H:\Basis_Functions_Butane\eps_sig_lam_all',skiprows=1)
sig_all = eps_sig_lam_all[:,4]
eps_all = eps_sig_lam_all[:,3]

f, axarr = plt.subplots(nrows=1,ncols=1,figsize=(10,10))  

axarr.plot(sig_all[prob_propane>0.005],eps_all[prob_propane>0.005],'bs',alpha=0.5)
axarr.plot(sig_all[prob_butane>0.005],eps_all[prob_butane>0.005],'g^',alpha=0.5)
axarr.plot(sig_Potoff,eps_Potoff,'kx',markersize=14,markeredgewidth=5,label=r'Potoff, $\lambda_{\rm CH_2} = 16$')
#axarr.plot(sig_ref,eps_ref,'bs',mfc='None',markersize=10,label='References (PCFR-optimal)')
axarr.set_xlabel(r'$\sigma_{\rm CH_2}$ (nm)',fontsize=18)
axarr.set_ylabel(r'$\epsilon_{\rm CH_2}$ (K)',fontsize=18)
axarr.set_xlim([0.39,0.405])
axarr.set_ylim([55,65])
#axarr.set_yticks([59,60,61,62,63])
#axarr.set_xticks([0.393,0.396,0.399,0.402])
axarr.legend(loc=2)
axarr.set_title(r'Acceptable')# \left(\frac{kg}{m^3}\right)$')

f.savefig('acceptable_propane_butane_octane.pdf')

plt.show()

accept_butane = prob_butane>0.005

Z_low = np.ones(19)*100.
Z_high = np.ones(19)*(-100.)

for i, accept in enumerate(accept_butane):
    if accept:
        UPZ = np.loadtxt('H:/Basis_Functions_Butane/MBAR_rr'+str(i),skiprows=1)
    
        USim = UPZ[:,0]
        ZSim = UPZ[:,4]
        PSim = UPZ[:,2]
        dPSim = UPZ[:,3]
        dZSim = ZSim * dPSim/PSim
        dUSim = UPZ[:,1]
        
        for iZ, Z in enumerate(ZSim):
            
            if Z < Z_low[iZ]: Z_low[iZ] = Z.copy()
            if Z > Z_high[iZ]: Z_high[iZ] = Z.copy()