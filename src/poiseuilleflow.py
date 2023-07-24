import numpy as np
import matplotlib.pyplot as plt
from seqLB.LB import LB  
from seqLB.boundaries import * 

import numpy as np
import matplotlib.pyplot as plt
from seqLB.LB import LB  
from seqLB.boundaries import * 

omega = 1
Nx = 50 
Ny = 50 

pIn = 0.31
pOut = 0.3
cs = 1/np.sqrt(3)
viscosity = (1/3)*((1/omega) - 0.5) 
y = np.arange(Ny)

timesteps = 1500
SAVE_EVERY = 100 
    
def simulate(lb):  
    # boundaries to apply to the simulation 
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
            lb.updateDensityFigure(lb.rho, timestep = i)
            lb.updateVelocityFigure(lb.ux, lb.uy, timestep = i)
            axes[0].cla()
            dynamic_viscosity = (lb.rho[:, lb.Nx//2] * viscosity)
            partial_derivative = (pOut - pIn) / lb.Nx
            analytical = (-0.5 * partial_derivative * y * (lb.Ny - 1 - y)) / dynamic_viscosity
            axes[0].plot(lb.ux[:, lb.Nx//2], y)
            axes[0].plot(analytical, y)
            axes[0].set_ylabel('y')
            axes[0].set_xlabel('velocity')
            axes[0].legend(['Simulated', 'Analytical'])
            figs[0].savefig(f"./plots/poiseuilleflow/img_{i}", bbox_inches='tight', pad_inches=0)

    
def poiseuille(): 

    lb = LB()
    lb.omega = omega
    lb.fitParams()

    simulate(lb)

if __name__ == '__main__':
    poiseuille()

