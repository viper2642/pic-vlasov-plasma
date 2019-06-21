
import numpy as np
import matplotlib.pyplot as plt

class Efield:
    def __init__(self,nintervals=100):
        self.nmesh=nintervals+1
        self.dx=1./nintervals
        self.grid=np.linspace(0.,1.,self.nmesh)
        self.potential=self.grid  # linear
        self.charge_density=np.zeros(self.nmesh)
        
    # linear interpolation
    def eval_pot(self,x):
        return np.interp(x,self.grid,self.potential)
    
    def eval_field(self,x):
        return np.interp(x,self.grid[:-1],np.diff(self.potential)/np.diff(self.grid))

    def eval_charge_density(self,x):
        return np.interp(x,self.grid,self.charge_density)

    def plot(self):
        plt.plot(self.grid,self.charge_density,'-xk',label='data')

        # check interpolation
        xfine=np.linspace(0.,1.,300.)
        yfine=self.eval_charge_density(xfine)
        plt.plot(xfine,yfine,'.-r',label='interp')

    def deposit(self,particles):
        ## bin the particles within the grid
        #charge,tmp=np.histogram(particles.pos(),bins=self.grid,weights=particles.w*particles.q/self.dx)
        #self.charge_density[:-1]+=charge

        for x,w in zip(particles.pos(),particles.w):
            ind=int(np.floor(x/self.dx))  # lower index
            percentage=x/self.dx-ind
            self.charge_density[ind]+=(1.-percentage)*w*particles.q/self.dx
            self.charge_density[ind+1]+=percentage*w*particles.q/self.dx
        # periodize
        self.charge_density[0]+=self.charge_density[-1]
        self.charge_density[-1]=self.charge_density[0]

        


class Species:
    def __init__(self,nparticles=100.,mass=1.,charge=1.,temperature=1.):
        self.n=nparticles
        self.T=temperature
        self.m=mass
        self.q=charge
        self.vth=np.sqrt(temperature/self.m)

        self.y=np.asarray([np.random.rand(self.n),self.vth*np.random.randn(self.n)]).T
        self.w=np.ones(self.n)/self.n

    def pos(self):
        return self.y[:,0]

    def vel(self):
        return self.y[:,1]

    def push(self,dt,efield):
        ## leap-frog
        #half-step
        self.y[:,1]+=0.5*dt*self.q/self.m*efield.eval_field(self.pos())
        self.y[:,0]+=dt*self.y[:,1]
        self.y[:,1]+=0.5*dt*self.q/self.m*efield.eval_field(self.pos())

        # periodise
        self.y[:,0]%=1.
    
        
        
    def plot(self):
        plt.plot(self.pos(),self.vel(),'.')

    
        
