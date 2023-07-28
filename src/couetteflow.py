import numpy as np
import matplotlib.pyplot as plt
from seqLB.LB import LB, PLOTDIR
from seqLB.boundaries import * 


# Grid size in x and y directions 
Nx = 50
Ny = 50

omega = 1

# velocity of the moving wall 
wallVelocity = 0.1 

timesteps = 4000
SAVE_EVERY = 100 
    
def simulate(lb):  

  # wall velocity and helper variables
  wvarr = [0.0, wallVelocity]
  y = np.arange(Ny)
  analyticalVelocity = (y / (Ny-1)) * wvarr[1] 
 
  # boundaries to apply to the simulation 
  boundaries = []
  boundaries.append(BottomWallBoundary(lb))
  boundaries.append(MovingTopWallBoundary(lb, wallVelocity))
 

  fig, ax = plt.subplots()
  fig2, ax2 = plt.subplots()
  figs, axes = [fig, fig2], [ax, ax2]

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
        lb.updateDensityFigure(lb.rho, timestep = i)
        lb.updateVelocityFigure(lb.ux, lb.uy, timestep = i)
        axes[0].cla()
        axes[0].set_xlim([-0.01, wallVelocity])
        axes[0].axhline(0.0, color='k')
        axes[0].axhline(lb.Ny-1, color='r')
        axes[0].plot(lb.ux[:,lb.Nx//2], y)
        axes[0].plot(analyticalVelocity, y)
        axes[0].set_ylabel('y')
        axes[0].set_xlabel('velocity')
        axes[0].legend(['Moving Wall', 'Rigid Wall',
                        'Simulated Velocity', 'Analytical Velocity'])
        figs[0].savefig(f"{PLOTDIR}/couetteflow/img_{i}", bbox_inches='tight', pad_inches=0)

def couetteflow(): 
    
    lb = LB()
    lb.Nx = Nx 
    lb.Ny = Ny 
    lb.omega = omega
    lb.fitParams()

    simulate(lb)

if __name__ == '__main__':
    couetteflow()

