import numpy as np
import matplotlib.pyplot as plt
from seqLB.LB import LB  
from seqLB.boundaries import * 

import numpy as np
import matplotlib.pyplot as plt
from seqLB.LB import LB, PLOTDIR
from seqLB.boundaries import * 

# grid size in x and y direction 
Nx = 100
Ny = 50
# omega parameter for this simulation 
omega = 0.7

# pressure in and out 
pIn = 0.31
pOut = 0.3
cs = 1/np.sqrt(3)
# analytical viscosity dependent on omega 
viscosity = (1/3)*((1/omega) - 0.5) 
y = np.arange(Ny)

# timesteps to simulate every SAVE_EVERYth simulation step is plotted 
timesteps = 5000
SAVE_EVERY = 100 
    
def simulate(lb):  
    # boundaries to apply to the simulation 
    # represent a pipe 
    boundaries = []
    boundaries.append(TopWallBoundary(lb))
    boundaries.append(BottomWallBoundary(lb))
    boundaries.append(HorizontalInletOutletBoundary(lb, Ny, pIn, pOut, cs))

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
            # plot analytical solution and simulated velocity
            lb.updateDensityFigure(lb.rho, timestep = i)
            lb.updateVelocityFigure(lb.ux, lb.uy, timestep = i)
            axes[0].cla()
            dynamicVis = (lb.rho[:, lb.Nx//2] * viscosity)
            partialDeriv = (pOut - pIn) / lb.Nx
            analytical = (-0.5 * partialDeriv * y * (lb.Ny - 1 - y)) / dynamicVis
            axes[0].plot(lb.ux[:, lb.Nx//2], y)
            axes[0].plot(analytical, y)
            axes[0].set_ylabel('y')
            axes[0].set_xlabel('velocity')
            axes[0].legend(['Simulated', 'Analytical'])
            figs[0].savefig(f"{PLOTDIR}/poiseuilleflow/img_{i}", bbox_inches='tight', pad_inches=0)

    
def poiseuille(): 

    lb = LB()
    lb.Nx = Nx 
    lb.Ny = Ny 
    lb.omega = omega
    lb.fitParams()

    simulate(lb)

if __name__ == '__main__':
    poiseuille()

