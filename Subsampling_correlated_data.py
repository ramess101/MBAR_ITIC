# -*- coding: utf-8 -*-
"""
Subsamples from the output data

@author: ram9
"""

from __future__ import division
import numpy as np 
import os, sys, argparse, shutil
from pymbar import MBAR, timeseries
import matplotlib.pyplot as plt

#Read in the simulation specifications

fpath_root = 'C:/calc1_Backup/Ethane/Gromacs/LJscan/ref0/'
fpath_end = '/NVT_eq/NVT_prod/energy_press_ref0rr0.xvg'

ITIC = np.array(['Isotherm', 'Isochore'])
nTemps = {'Isochore':2,'Isotherm':1}
nrhos = {'Isochore':5,'Isotherm':9}

# Create a list of all the file paths (without the reference directory, just the run_type, rho, Temp)

fpath_all = []
state_points = []

for run_type in ITIC: 

    for irho  in np.arange(0,nrhos[run_type]):

        for iTemp in np.arange(0,nTemps[run_type]):

            if run_type == 'Isochore':

                fpath_all.append(fpath_root+run_type+'/rho'+str(irho)+'/T'+str(iTemp)+fpath_end)
                state_points.append('IC_rho'+str(irho)+'T'+str(iTemp))

            else:

                fpath_all.append(fpath_root+run_type+'/rho_'+str(irho)+fpath_end)
                state_points.append('IT_rho'+str(irho))
                
iSubLJ_all = np.zeros(len(fpath_all))
LJSubave = np.zeros(len(fpath_all))
LJave = np.zeros(len(fpath_all))
iSubP_all = np.zeros(len(fpath_all))
PSubave = np.zeros(len(fpath_all))
Pave = np.zeros(len(fpath_all))
                
for ipath, fpath in enumerate(fpath_all):
    
    en_p = open(fpath).readlines()[28:]
    
    nSnapsRef = len(en_p)
    t=np.zeros([nSnapsRef])
    LJsr=np.zeros([nSnapsRef])
    Psr=np.zeros([nSnapsRef])

    for frame in xrange(nSnapsRef):
        t[frame]=float(en_p[frame].split()[0])
        LJsr[frame] = float(en_p[frame].split()[1])
        Psr[frame] = float(en_p[frame].split()[5])
    
    iSubLJ = pymbar.timeseries.subsampleCorrelatedData(LJsr)
    iSubLJ_all[ipath] = len(iSubLJ)
    
    LJave[ipath] = np.mean(LJsr)
    LJSubave[ipath] = np.mean(LJsr[iSubLJ])
    
    iSubP = pymbar.timeseries.subsampleCorrelatedData(Psr)
    iSubP_all[ipath] = len(iSubP)
    
    Pave[ipath] = np.mean(Psr)
    PSubave[ipath] = np.mean(Psr[iSubP])
            
plt.plot(iSubLJ_all)
plt.plot(iSubP_all)
plt.xlabel('State Point')
plt.ylabel('Number of Uncorrelated Samples')
plt.show()

fig, ax = plt.subplots(figsize=(16,12))
bars1 = ax.bar(np.arange(19)-0.15,iSubLJ_all,0.3,label='LJ Energy')
bars2 = ax.bar(np.arange(19)+0.15,iSubP_all,0.3,label='Pressure')
ax.set_xlabel('State Points')
ax.set_ylabel('Number of Uncorrelated Samples')
ax.set_xticks(range(19))
ax.set_xticklabels(state_points)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.legend()
plt.show()

plt.plot(LJave)
plt.plot(LJSubave)
plt.show()

devLJ = (LJSubave - LJave)/LJave*100.
        
plt.plot(devLJ)
plt.xlabel('State Point')
plt.ylabel('Percent Deviation in LJ')
plt.show()

devP = PSubave - Pave
        
plt.plot(devP)
plt.xlabel('State Point')
plt.ylabel('Deviation in P (bar)')
plt.show()