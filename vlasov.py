"""
Class definitions Vlasov-Poisson PIC simulation

author: David Pfefferlé
email: david.pfefferle@uwa.edu.au
website: http://viper2642.github.com
license: GPL-3.0
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np

class Efield:
    """ Representation of the electric field

    Attributes
    ----------
    t : time
    V : Dirichlet potential applied on the right edge
    omega : frequency of alternating cosine Dirichlet potential (cosine)
    nmesh : number of grid points
    dx : grid spacing
    grid : grid points [0,1]
    half_grid : half-mesh grid points (for gradient of potential)
    potential : value of electric potential on grid points
    charge_density : value of charge density on grid points
    efield : value of gradient of potential on half-mesh points 
    
    """
        
    def __init__(self,nintervals=100,voltage=0.,frequency=0.):
        """ 
        Parameters
        ----------
        nintervals : int
           number of intervals in the grid (nmesh=nintervals+1)
        voltage : float (default 0.)
           Dirichlet potential applied on the right edge
        frequency : float (default 0.)
           frequency of alternating cosine Dirichlet potential
        """
        
        self.t=0.
        self.V=voltage
        self.omega=frequency
        
        self.nmesh=nintervals+1
        self.dx=1./nintervals
        self.grid=np.linspace(0.,1.,self.nmesh)
        self.half_grid=self.grid[:-1]+0.5*self.dx
        self.potential=np.zeros(self.nmesh)
        self.charge_density=np.zeros(self.nmesh)
        self.efield=np.zeros(nintervals)
        
    def eval_potential(self,x):
        """ evaluate the potential at x using linear interpolation"""
        return np.interp(x,self.grid,self.potential)

    def eval_field(self,x):
        """ evaluate the electric field at x using linear interpolation of the gradient (half mesh)"""
        return np.interp(x,self.half_grid,self.efield)        

    def eval_charge_density(self,x):
        """ evaluate the charge density at x using linear interpolation"""
        return np.interp(x,self.grid,self.charge_density)

    def deposit(self,particles):
        """ deposit the charge from a particle type on the mesh points
        
        distribute charge on left and right grid point proportionally to distance
        """
        for x,w in zip(particles.pos(),particles.w):
            ind=int(np.floor(x/self.dx))  # lower index
            percentage=x/self.dx-ind      # distance from points
            self.charge_density[ind]+=(1.-percentage)*w*particles.q/self.dx
            self.charge_density[ind+1]+=percentage*w*particles.q/self.dx


    def solve(self):
        """ Solve the Poisson equation on mesh points

        Finite difference of d^2 P /dx^2 = rho becomes [l, g, u] P = rho 
        where tridiagonal matrix with g = -2 diagonal, u = 1 upper diagonal, l = 1 lower diagonal
        right-hand side vector rho. 
        
        Thomas algorithm is used to invert tridiagonal matrix.
        """
        
        N=self.nmesh-2

        #Thomas algorithm        
        u=np.ones(N-1)
        l=np.ones(N-1)
        g=-2.*np.ones(N)
        rho=self.dx**2*np.copy(self.charge_density[1:-1])
        
        # Dirichlet boundary condition on the right
        self.potential[-1]=self.V*np.cos(self.t*self.omega)
        rho[-1]-=self.potential[-1]
        
        #initial step
        u[0] = u[0]/g[0]
        rho[0] = rho[0]/g[0]
        # first pass
        for i in range(1,N-1):
            u[i] = u[i]/(g[i]-u[i-1]*l[i-1])
        # second pass 
        for i in range(1,N):
            rho[i] = (rho[i]-rho[i-1]*l[i-1])/(g[i]-u[i-1]*l[i-1])
        # solution on the penultimate grid point
        self.potential[-2] = rho[-1]
        # propagate solution backwards
        for i in range(N-2,-1,-1):
            self.potential[i+1] = rho[i] - u[i]*self.potential[i+2]

        # compute gradient of potential on half-mesh
        self.update_gradient()
        

    def update_gradient(self):
        """ update the gradient of potential on half-mesh points. """

        self.efield=np.diff(self.potential)/self.dx

        
    def push(self,dt):
        """ advance time by dt. """
        self.t+=dt


class Species:
    """ representation of a particle species 
    Attributes
    ----------
    n : number of markers
    T : initial temperature (Gaussian) in velocity space
    m : species mass
    q : species charge
    vth : initial thermal velocity
    y : position and velocity array
    w : particle weight
    
    """
    
    def __init__(self,nparticles=100,mass=1.,charge=1.,temperature=1.):
        """" 
        Parameters
        ----------
        nparticles : int 
             number of particles
        mass : float
             species mass
        charge : float
             species charge
        temperature : float
             species initial temperature as Gaussian distribution of velocities
        """
        
        self.n=nparticles
        self.T=temperature
        self.m=mass
        self.q=charge
        self.vth=np.sqrt(temperature/self.m)

        self.y=np.asarray([np.random.rand(self.n),self.vth*np.random.randn(self.n)]).T
        self.w=np.ones(self.n)/self.n

    def pos(self):
        """ position of particles """
        return self.y[:,0]

    def vel(self):
        """ velocity of particles """
        return self.y[:,1]
    
    def push(self,dt,efield):
        """Push particles over time dt given electric field efield

        Particle push is performed using 2nd order leap-frog
        algorithm. Particles that leave the domain are reinjected on
        the other side (periodic domain)

        """
        self.y[:,1]+=0.5*dt*self.q/self.m*efield.eval_field(self.pos())
        self.y[:,0]+=dt*self.y[:,1]
        self.y[:,1]+=0.5*dt*self.q/self.m*efield.eval_field(self.pos())

        # periodise
        self.y[:,0]%=1.
    
    
class Plasma:
    """representation of a plasma as an electric field, ion and electrons species"""
    
    def __init__(self,ions=Species(1000,mass=1.,charge=1.,temperature=1.),electrons=Species(1000,mass=1./10.,charge=-1.,temperature=1.),electric_field=Efield()):
        self.E=electric_field
        self.ion=ions
        self.ele=electrons
        
        self.E.deposit(self.ion)
        self.E.deposit(self.ele)
        self.E.solve()
    
    def evolve(self,dt):
        """Evolve plasma by dt

        First push the particles, then deposit the charges and finally
        solve for electric field (and push in time)

        """
        self.ion.push(dt,self.E)
        self.ele.push(dt,self.E)
        self.E.deposit(self.ion)
        self.E.deposit(self.ele)
        self.E.solve()
        self.E.push(dt)

        
    def particle_number(self):
        """ total particle number in the plasma."""
        
        return np.sum(self.ion.w)+np.sum(self.ele.w)

    def global_charge(self):
        """ total charge in the plasma."""
        return self.ion.q*np.sum(self.ion.w)+self.ele.q*np.sum(self.ele.w)

    def current(self):
        """ total current in the plasma."""
        
        return self.ion.q*np.sum(self.ion.w*self.ion.vel())+self.ele.q*np.sum(self.ele.w*self.ele.vel())
    
