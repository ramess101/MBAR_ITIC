
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize  

class ITIC_VLE(object):
    
    def __init__(self, Tsat, rhol, rhov, Psat):
        
        # Ensures that the lowest density IC corresponds to maximum temperature 
        rhol = rhol[np.argmax(Tsat):]
        rhov = rhov[np.argmax(Tsat):]
        Psat = Psat[np.argmax(Tsat):]
        Tsat = Tsat[np.argmax(Tsat):]  
    
        # Ensures that no Psats of 0
        Tsat = Tsat[Psat>0]
        rhol = rhol[Psat>0]
        rhov = rhov[Psat>0]
        Psat = Psat[Psat>0]
        
        self.Tsat = Tsat
        self.rhol = rhol
        self.rhov = rhov
        self.Psat = Psat
        self.logPsat = np.log10(Psat)
        self.invTsat = 1000./Tsat
        self.boptPsat = np.zeros(3)
        self.boptrhol = np.zeros(4)
        
    def SSE(self,data,model):
        SE = (data - model)**2
        SSE = np.sum(SE)
        return SSE 

    def logPAntoine(self,b,T):
        logP = b[0] - b[1]/(b[2] + T)
        return logP

    def guessAntoine(self):
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.invTsat,self.logPsat)
        guess = np.array([intercept,-slope*1000.,0])
        return guess
    
    def fitAntoine(self):
        Tfit = self.Tsat
        logPfit = self.logPsat
        SSElog = lambda b: self.SSE(logPfit,self.logPAntoine(b,Tfit))
        guess = self.guessAntoine()
        if len(Tfit) >= 3:
            bopt = minimize(SSElog,guess).x
        else:
            bopt = guess
        self.boptPsat = bopt
        return bopt
    
    def logPsatHat(self,T):
        if np.all(self.boptPsat == np.zeros(3)):
            self.fitAntoine()
        bopt = self.boptPsat
        logPsatHat = self.logPAntoine(bopt,T)
        return logPsatHat

    def PsatHat(self,T):
        PsatHat = 10.**(self.logPsatHat(T))
        return PsatHat
    
    def rholRectScale(self,b,T):
        beta = 0.326
        rhol = b[0] + b[1]*(b[2] - T) + b[3]*(b[2] - T)**beta
        return rhol
    
    def guessRectScale(self):
        rhor = (self.rhol + self.rhov)/2.
        rhocGuess = np.mean(rhor)
        TcGuess = np.max(self.Tsat)/0.85
        guess = np.array([rhocGuess,2,TcGuess,50])
        ydata = self.rhol - rhocGuess #Modify to get decent guesses
        xhat = lambda b: self.rholRectScale(np.array([rhocGuess,b[0],TcGuess,b[1]]),self.Tsat) - rhocGuess
        SSEguess = lambda b: self.SSE(ydata,xhat(b))
        guess = np.array([2,50])
        bnd = ((0,None),(0,None))
        bopt = minimize(SSEguess,guess,bounds=bnd).x
        guess = np.array([rhocGuess,bopt[0],TcGuess,bopt[1]])
        return guess
    
    def fitrhol(self):
        Tfit, rholfit = self.Tsat, self.rhol
        SSErhol = lambda b: self.SSE(rholfit,self.rholRectScale(b,Tfit))
        guess = self.guessRectScale()
        #print(SSErhol(guess))
        if len(Tfit) >= 4:
            bnd = ((0,np.min(rholfit)),(0,None),(np.max(Tfit),None),(0,None))
            bopt = minimize(SSErhol,guess,bounds=bnd).x
        else:
            bopt = guess
        self.boptrhol = bopt
        #bopt = guess #If the optimization is not converging, this is a better option
        return bopt
    
    def rholHat(self,T):
        if np.all(self.boptrhol == np.zeros(4)):
            self.fitrhol()
        bopt = self.boptrhol
        rholHat = self.rholRectScale(bopt,T)
        return rholHat

def main():
    
    # Values for Potoff model predicted from TraPPE samples
    # Taken from PCFR-ITIC
    Tsat = np.array([276.778, 251.3329, 221.4212, 187.5534, 149.2056])
    rhol = np.array([428.5761,471.4212,514.2969,557.1258,600.0072])
    rhov = np.array([43.59846,21.13519,7.967988,1.86442,0.143268])
    Psat = np.array([23.62316,12.02041,4.435127,0.938118,0.058897])
    
    # Taken from MBAR-ITIC
#    Tsat = np.array([211.3574, 197.6162, 178.8908, 123.1425, 102.5166])
#    rhol = np.array([428.5761,471.4212,514.2969,557.1258,600.0072])
#    rhov = np.array([10.07994, 7.19709, 3.250055, 0.042836, 0.002265])
#    Psat = np.array([30.20705, 19.92031, 7.845697, 0.12644, 0.007994])
##    

    #ref0rr21 for scan at highEps with lam16. Perhaps IC is above critical point
#    Tsat = np.array([38.4551842,	152.0941454,	102.5972211,	68.34082747,	64.65333992])
#    rhol = np.array([428.5760696,	471.4212141,	514.2968664,	557.1257821,	600.0072033])
#    rhov = np.array([0,	0.514123698,	0.100480245,	4.62E-05,	1.33E-05])
#    Psat = np.array([0,	1.265481573,	0.376355903,	0.000244417,	7.41E-05])
    
    #ref0rr123 problematic for fitting because of bounds on guess
#    Tsat = np.array([95.42869923,	165.489684,	136.1076539,	81.68891776,	75.7025512])
#    rhol = np.array([428.5760696,	471.4212141,	514.2968664,	557.1257821,	600.0072033])
#    rhov = np.array([0.012913259,	2.580919739,	1.082752784,	3.55E-04,	7.02E-05])
#    Psat = np.array([0.04927028,	6.977463115,	3.689933732,	0.001572231,	3.35E-04])
    
    #ref0rr168 problematic for fitting, overflow in Psat
#    Tsat = np.array([38.68872081,	152.57235,	104.491397,	68.76362042,	64.86466136])
#    rhol = np.array([428.5760696,	471.4212141,	514.2968664,	557.1257821,	600.0072033])
#    rhov = np.array([0,	0.742605243,	0.089549641,	2.29E-05,	5.69E-06])
#    Psat = np.array([0,	1.857142769,	0.321819648,	0.000120528,	3.17E-05])

    ITIC_fit = ITIC_VLE(Tsat,rhol,rhov,Psat)
    
    invTsat = ITIC_fit.invTsat
    logPsat = ITIC_fit.logPsat
    Tsat = ITIC_fit.Tsat
    rhol = ITIC_fit.rhol
    rhov = ITIC_fit.rhov
    Psat = ITIC_fit.Psat
    
#    invTsat = 1000./Tsat
#    logPsat = np.log10(Psat)

    Tplot = np.linspace(min(Tsat),max(Tsat),1000)
    invTplot = 1000./Tplot
    logPsatplot = ITIC_fit.logPsatHat(Tplot)
    rholplot = ITIC_fit.rholHat(Tplot)
    Psatplot = ITIC_fit.PsatHat(Tplot)
    Psatsmoothed = ITIC_fit.PsatHat(Tsat)
    rholsmoothed = ITIC_fit.rholHat(Tsat)
    
    print(Psatplot)
    print(Psatsmoothed)
    
    plt.plot(invTsat,logPsat,'ro')
    plt.plot(invTplot,logPsatplot,'k')
    plt.xlabel('1000/T (K)')
    plt.ylabel('log(Psat/bar)')
    plt.show()
    
    plt.plot(Tsat,logPsat,'ro')
    plt.plot(Tplot,logPsatplot,'k')
    plt.xlabel('Temperature (K)')
    plt.ylabel('log(Psat/bar)')
    plt.show()
    
    plt.plot(Tsat,Psat,'ro')
    plt.plot(Tsat,Psatsmoothed,'gx')
    plt.plot(Tplot,Psatplot,'k')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Psat (bar)')
    plt.show()

    plt.plot(rhol,Tsat,'ro')
    plt.plot(rhov,Tsat,'ro')
    plt.plot(rholsmoothed,Tsat,'gx')
    plt.plot(rholplot,Tplot,'k')
    plt.xlabel('Density (kg/m3)')
    plt.ylabel('Temperature (K)')
    plt.show()

if __name__ == '__main__':
    
    main()