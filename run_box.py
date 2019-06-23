#!/usr/bin/python

"""
Driver for brute-force plasma box simulation

author: David Pfefferlé
email: david.pfefferle@uwa.edu.au
website: http://viper2642.github.com
license: GPL-3.0
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from plasma_box import Plasma

## simulation parameter
npart=10
plasma_temperature=5.
dt=2.e-2
tfin=1.
waittime=30 # ms

## plasma initialisation
plas=Plasma(npart,temperature=plasma_temperature,electron_mass_ratio=1./5.,rtol=1.e-5)

## cosmetics
plt.rcParams.update({'font.size': 16})
fig, alist = plt.subplots(1,2,figsize=(16,8))
fig.subplots_adjust(left=0.05, bottom=0.08, right=0.99, top=0.99, wspace=None, hspace=None)
# plasma box 
alist[0].set_xlim(0, 1)
alist[0].set_ylim(0, 1)
alist[0].set_aspect('equal', adjustable='box')
alist[0].set_xlabel('x')
alist[0].set_ylabel('y')
# energy recording
alist[1].set_autoscale_on
alist[1].set_xlabel('t')
alist[1].set_ylabel('energy')
alist[1].set_yticklabels([])
alist[1].set_xticklabels([])
alist[1].legend(loc="lower right")

## plot initialisation
elec, = alist[0].plot([],[],'.b',markersize=10)
ions, = alist[0].plot([],[],'.r',markersize=20)
kin,  = alist[1].plot([],[],'-r',label='kinetic')
pot,  = alist[1].plot([],[],'-b',label='potential')
erg,  = alist[1].plot([],[],'-k',label='total')

plist=(elec,ions,kin,pot,erg)

# initial state of plot
def init():
    for p in plist:
        p.set_data([],[])
    return plist

# animation function
def animate(i):
    global dt, fig, alist, plas
    
    plas.push(dt)
        
    x,y=plas.get_electrons()
    elec.set_data(x,y)
    x,y=plas.get_ions()
    ions.set_data(x,y)
  
    erg.set_xdata(np.append(erg.get_xdata(),plas.t))
    erg.set_ydata(np.append(erg.get_ydata(),plas.toterg()-plas.T0))
    
    kin.set_xdata(np.append(kin.get_xdata(),plas.t))
    kin.set_ydata(np.append(kin.get_ydata(),plas.kinerg()-plas.K0))
    
    pot.set_xdata(np.append(pot.get_xdata(),plas.t))
    pot.set_ydata(np.append(pot.get_ydata(),plas.poterg()-plas.V0))
    
    alist[1].relim()
    alist[1].autoscale_view()
             
    return plist

ani = anim.FuncAnimation(fig,animate,interval=waittime,blit=True,init_func=init)
    
plt.show()
