"""
Class definitions for brute-force plasma box simulation

author: David Pfefferlé
email: david.pfefferle@uwa.edu.au
website: http://viper2642.github.com
license: GPL-3.0
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from scipy.spatial.distance import pdist,squareform

class Plasma:
    """ representation of the plasma as a collection of charged particles (ions and electrons)
    
    Attributes
    ----------
    n : number of particles per species
    T : initial temperature of species (Gaussian velocity distribution)
    q : array of charges (upper half ion, lower half electron)
    m : array of masses (upper half ion, lower half electron)
    vth : array of initial thermal velocities (upper half ion, lower half electron)
    y :  phase-space coordinates (2n,4) with x,y,vx,vy
         initialised uniformly in position and bi-gaussian in velocity space
    k : coulomb force proportionality constant
    t : time
    rtol : ODE solver relative tolerance
    """
    def __init__(self,nparticles=10,temperature=1.,electron_mass_ratio=1./5.,rtol=1.e-6):
        self.n = int(nparticles)
        self.T = temperature
        
        # charge ion, charge electron
        self.q = np.append(np.ones(self.n),-np.ones(self.n))
         # mass ion, mass electron
        self.m = np.append(np.ones(self.n),electron_mass_ratio*np.ones(self.n))
        self.vth = np.asarray([np.sqrt(self.T/self.m),np.sqrt(self.T/self.m)]).T
        # phase-space variables
        self.y = np.concatenate((np.random.rand(2*self.n,2),self.vth*np.random.randn(2*self.n,2)),axis=1)
        
        self.k=0.01  # coulomb coupling
        self.t=0.    # time
        
        # for calculating the force
        self.chrgmat=self.q.reshape((1, -1,1))*self.q.reshape((-1, 1,1))
        self.mssmat=np.asarray([self.m,self.m]).T
        # for calculating the potentiel energy
        self.chrgcoup=self.q.reshape((1,-1))*self.q.reshape((-1,1))
        np.fill_diagonal(self.chrgcoup,0.)
        self.chrgcoup=squareform(self.chrgcoup)
        
        # initial energies
        self.K0=self.kinerg()
        self.V0=self.poterg()
        self.T0=self.toterg()

        self.rtol=rtol
    
    
    def backinbox(self):
        """ mirror particles off wall """

        # check whether particles have run out of bounds
        crossed_x1 = self.y[:,0] < 0
        crossed_x2 = self.y[:,0] > 1
        crossed_y1 = self.y[:,1] < 0
        crossed_y2 = self.y[:,1] > 1

        # mirror velocity
        self.y[crossed_x1 | crossed_x2, 2] *= -1.
        self.y[crossed_y1 | crossed_y2, 3] *= -1.

        
    def kinerg(self):
        """ total kinetic energy of ions and electrons. """
        return 0.5*np.sum(self.m*np.sum(self.y[:,2:]**2,axis=1))
    
    def poterg(self):
        """ total potential (interaction) energy of ions and electrons. """        
        return self.k*np.sum(self.chrgcoup/pdist(self.y[:,:2]))
    
    def toterg(self):
        """ total energy (potential+kinetic)."""
        return self.kinerg()+self.poterg()
    
    def push(self,dt):
        """ push particles by dt using scipy solve_ivp """
        stepper=solve_ivp(self.accel,(self.t,self.t+dt),self.y.flatten(),rtol=self.rtol)
        self.y=stepper.y[:,-1].reshape(-1,4)
        self.t=stepper.t[-1]
        
        self.backinbox()
        
    def accel(self,time,y):
        """ array of acceleration for ions and electrons from Coulomb interaction. """
        ytmp=y.reshape(-1,4) # magic command to recover the data in x,y,vx,vy
        acc=np.concatenate((ytmp[:,2:],self.force(ytmp[:,:2])/self.mssmat),axis=1)
        return acc.flatten()
    
    def force(self,pos):
        """ array of Coulomb force. """
        # 2d displacement matrix
        disps=pos.reshape((1,-1,2))-pos.reshape((-1,1,2))
        # distance array
        dists=norm(disps,axis=2)
        dists[dists==0]=1.  # Avoid divide by zero warnings
        # Coulomb force array
        return -self.k*(disps*self.chrgmat/np.expand_dims(dists,2)**3).sum(axis=1)
        
    def get_electrons(self):
        """ 2D position of electrons."""
        return self.y[self.n:,0],self.y[self.n:,1]
    
    def get_electron_velocity(self):
        """ 2D velocity of electrons."""
        return self.y[self.n:,2],self.y[self.n:,3]
    
    def get_ions(self):
        """ 2D position of ions."""
        return self.y[:self.n,0],self.y[:self.n,1]
    
    def get_ion_velocity(self):
        """ 2D velocity of ions."""
        return self.y[:self.n,2],self.y[:self.n,3]
