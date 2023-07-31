import numpy as np
import matplotlib.pyplot as plt
from parallelLB.LB import LB, PLOTDIR
from parallelLB.boundaries import * 
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD

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

timesteps = 10000

# enable plotting
PLOTTING = False 
SAVE_EVERY = 1000

# velocity vectors, splitted into x and y components
cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])

# stuff for plotting  
xaxis = np.linspace(0, Nx, Nx)
yaxis = np.linspace(0, Ny, Ny)
X, Y = np.meshgrid(xaxis, yaxis)
densityFig, densityAx = plt.subplots()
densityFigureMesh = densityAx.pcolormesh(X, Y, np.zeros((Ny,Nx)), shading='auto')
velocityFig, velocityAx = plt.subplots()
velocityStreamPlot = velocityAx.streamplot(X, Y, np.zeros((Ny,Nx)), np.zeros((Ny,Nx)), density = 2)


def simulate(lb):  

  # boundaries to apply to the simulation 
  boundaries = []
  boundaries.append(BottomWallBoundary(lb))
  boundaries.append(MovingTopWallBoundary(lb, wallVelocity))
  boundaries.append(RightWallBoundary(lb))
  boundaries.append(LeftWallBoundary(lb))

  timeBeforeSim = time.time()

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

    # communicate with neighboring processes to update ghost region 
    lb.communicateNonBlocking()
    #lb.gather()
    #lb.split()

    for boundary in boundaries: 
      boundary.after()

    if i % SAVE_EVERY == 0 and comm.Get_rank() == 0 and PLOTTING:          
        rho = np.sum(lb.F,-1)
        ux = np.sum(lb.F*cxs,2) / rho
        uy = np.sum(lb.F*cys,2) / rho
        densityAx.set_title(f"timestep: {i}")
        densityFigureMesh.update({'array':rho})
        densityFig.savefig(f'{PLOTDIR}/slidinglidParallel/density_timestep{i}.png')
        velocityAx.set_title(f"timestep: {i}")
        velocityAx.cla()
        velocityAx.streamplot(X, Y, ux, uy, density = 2)  
        velocityFig.savefig(f'{PLOTDIR}/slidinglidParallel/velocity_timestep{i}.png')

  return Nx * Ny * timesteps / (time.time() - timeBeforeSim) / 1e6

def slidinglid(): 
    lb = LB()
    lb.Nx = Nx
    lb.Ny = Ny 
    lb.omega = omega 
    lb.fitParams()
    mlups = simulate(lb)
    return mlups 


if __name__ == '__main__':
    start = time.time()
    mlups = slidinglid()
    if comm.Get_rank() == 0: 
      print(f"Execution time: {time.time() - start}s MLUPS {mlups}")