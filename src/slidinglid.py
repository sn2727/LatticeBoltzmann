import numpy as np
import matplotlib.pyplot as plt
from seqLB.LB import LB, PLOTDIR
from seqLB.boundaries import * 

# grid size in x and y direction
Nx = 300
Ny = 300
# omega parameter
omega = 1

# wall velocity
wallVelocity = 0.1

analyticalViscosity = (1/3)*((1/omega) - 0.5)  

# reynolds number 
re = (Nx*wallVelocity) / analyticalViscosity

timesteps = 100001
SAVE_EVERY = 1000
    

def simulate(lb):  

  # boundaries to apply to the simulation 
  boundaries = []
  boundaries.append(BottomWallBoundary(lb))
  boundaries.append(MovingTopWallBoundary(lb, wallVelocity))
  boundaries.append(RightWallBoundary(lb))
  boundaries.append(LeftWallBoundary(lb))

  for i in range(timesteps):

    # boundaries caching values before streaming 
    for boundary in boundaries: 
      boundary.before()

    # apply drift/stream
    lb.streaming()

    # Recalc local variables 
    lb.calculateDensity()
    lb.calculateVelocity()

    # apply collision
    lb.collision()

    for boundary in boundaries: 
      boundary.after()
  
    if i % SAVE_EVERY == 0:          
        lb.updateVelocityFigure(lb.ux, lb.uy, timestep = i)
        lb.velocityFig.density = 2 
        lb.velocityFig.savefig(f"{PLOTDIR}/slidinglid/img_{i}", bbox_inches='tight', pad_inches=0)


def slidinglid(): 
    lb = LB()
    lb.Nx = Nx
    lb.Ny = Ny 
    lb.omega = omega
    lb.fitParams()

    simulate(lb)

if __name__ == '__main__':
    slidinglid()

