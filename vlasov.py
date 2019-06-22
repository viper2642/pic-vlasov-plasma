
import numpy as np
import matplotlib.pyplot as plt

class Efield:
    def __init__(self,nintervals=100,voltage=0.):
        self.nmesh=nintervals+1
        self.dx=1./nintervals
        self.grid=np.linspace(0.,1.,self.nmesh)
        self.half_grid=self.grid[:-1]+0.5*self.dx
        self.potential=np.zeros(self.nmesh)
        self.charge_density=np.zeros(self.nmesh)
        self.efield=np.zeros(nintervals)
        self.potential[-1]=voltage
        
    # linear interpolation
    def eval_potential(self,x):
        return np.interp(x,self.grid,self.potential)

    # 
    def eval_field(self,x):
        return np.interp(x,self.half_grid,self.efield)        

    # linear interpolation
    def eval_charge_density(self,x):
        return np.interp(x,self.grid,self.charge_density)

    def plot(self):
        plt.plot(self.grid,self.charge_density,'-xb',label='charge density')
        plt.plot(self.grid,self.potential,'-xk',label='potential')
        plt.plot(self.grid,self.eval_field(self.grid),'-xr',label='electric field')
        plt.legend()

    # 
    def deposit(self,particles):
        # distribute charge on left and right grid point proportionally to distance
        for x,w in zip(particles.pos(),particles.w):
            ind=int(np.floor(x/self.dx))  # lower index
            percentage=x/self.dx-ind
            self.charge_density[ind]+=(1.-percentage)*w*particles.q/self.dx
            self.charge_density[ind+1]+=percentage*w*particles.q/self.dx
        
        # collect first and last and make them the same (periodic)
        # self.charge_density[0]+=self.charge_density[-1]
        # self.charge_density[-1]=self.charge_density[0]

    def solve(self):
        #Thomas algorithm
        N=self.nmesh-2

        u=np.ones(N-1)
        l=np.ones(N-1)
        g=-2.*np.ones(N)
        v=self.dx**2*np.copy(self.charge_density[1:-1])

        # Dirichlet boundary condition on the right
        v[-1]-=self.potential[-1]
        
        #initial step
        u[0] = u[0]/g[0]
        v[0] = v[0]/g[0]

        # first pass
        for i in range(1,N-1):
            u[i] = u[i]/(g[i]-u[i-1]*l[i-1])

        # second pass 
        for i in range(1,N):
            v[i] = (v[i]-v[i-1]*l[i-1])/(g[i]-u[i-1]*l[i-1])

        # solution on the penultimate grid point
        self.potential[-2] = v[-1]

        # propagate solution backwards
        for i in range(N-2,-1,-1):
            self.potential[i+1] = v[i] - u[i]*self.potential[i+2]

        self.update_gradient()

    # finite difference
    def update_gradient(self):
        self.efield=np.diff(self.potential)/self.dx

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

    
        
