"""
Creates tabulated .xvg files for gromacs to use
"""
from __future__ import division
import numpy as np 
import argparse

def create_tab(nrep,natt=6.,ncoul=1.):
    r = np.arange(0,2.401,0.0005)
    f = open('it_tab.xvg','w')
    rcutin = 0.05

    for ri in r:
        if ri < rcutin:
            U1 = rcutin**-ncoul
            F1 = ncoul * rcutin ** -(ncoul + 1.)
            U2 = -rcutin**-natt
            F2 = -natt * rcutin ** -(natt + 1.)
            U3 = rcutin**-nrep
            F3 = nrep * rcutin ** -(nrep + 1.)
        else:
            U1 = ri**-ncoul
            F1 = ncoul * ri ** -(ncoul + 1.)
            U2 = -ri**-natt
            F2 = -natt * ri ** -(natt + 1.)
            U3 = ri**-nrep
            F3 = nrep * ri ** -(nrep + 1.)
            
        f.write(str(ri)+'\t'+str(U1)+'\t'+str(F1)+'\t'+str(U2)+'\t'+str(F2)+'\t'+str(U3)+'\t'+str(F3)+'\n')
    f.close()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--lam",type=float,help="Set the value for lambda")
    args = parser.parse_args()
    if args.lam:
        create_tab(args.lam)
    else:
        print('Please specify a value for lambda')

if __name__ == '__main__':
    
    main()
