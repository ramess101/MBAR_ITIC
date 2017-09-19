
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, fsolve   

class ITIC_results(object):
    
    def __init__(self, Tsat, rhol, rhov, Psat):
        self.Tsat = Tsat
        self.rhol = rhol
        self.rhov = rhov
        self.Psat = Psat
        self.logPsat = np.log10(Psat)
        self.invTsat = 1000./Tsat
        
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
        bopt = minimize(SSElog,guess).x
        return bopt
    
    def logPsatHat(self,T):
        bopt = self.fitAntoine()
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
        bopt = minimize(SSEguess,guess).x
        guess = np.array([rhocGuess,bopt[0],TcGuess,bopt[1]])
        return guess
    
    def fitrhol(self):
        Tfit, rholfit = self.Tsat, self.rhol
        SSErhol = lambda b: self.SSE(rholfit,self.rholRectScale(b,Tfit))
        guess = self.guessRectScale()
        print(guess)
        print(SSErhol(guess))
        #bopt = minimize(SSErhol,guess).x
        bopt = guess #If the optimization is not converging, this is a better option
        return bopt
    
    def rholHat(self,T):
        bopt = self.fitrhol()
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
#    
    PCFR_ITIC_Potoff = ITIC_results(Tsat,rhol,rhov,Psat)
    
    invTsat = PCFR_ITIC_Potoff.invTsat
    logPsat = PCFR_ITIC_Potoff.logPsat

    Tplot = np.linspace(min(Tsat),max(Tsat),1000)
    invTplot = 1000./Tplot
    logPsatplot = PCFR_ITIC_Potoff.logPsatHat(Tplot)
    rholplot = PCFR_ITIC_Potoff.rholHat(Tplot)
    
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

    plt.plot(rhol,Tsat,'rx')
    plt.plot(rhov,Tsat,'rx')
    plt.plot(rholplot,Tplot,'k')
    plt.xlabel('Density (kg/m3)')
    plt.ylabel('Temperature (K)')
    plt.show()

if __name__ == '__main__':
    
    main()