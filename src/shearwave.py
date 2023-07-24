import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema 
from seqLB.LB import LB  

# Grid size in x and y directions 
Nx = 50
Ny = 50

# an initial average density used for initialization
rhoInitial = 1

# timesteps to simulate 
timesteps = 2000

# set the respective to true to either run the velocity or density experiment 
VELOCITY = True  
DENSITY = False 
assert(VELOCITY != DENSITY)

omega = 1 

# epsilon used for initializing the sinusoidal wave
epsilon = 0.01

# save every -th plot to src/plots/shearwave/
SAVE_EVERY = 50

def decay_perturbation(t, viscosity):
        size = Ny if VELOCITY else Nx
        return epsilon * np.exp(-viscosity * (2*np.pi/size)**2 * t)

def simulate(lb,  timesteps = 1000, plotting=True):
    analyticalViscosity = (1/3)*((1/lb.omega) - 0.5)  
    q = []
    for i in range(timesteps):

        # apply drift/stream
        lb.streaming()

        # Recalc local variables 
        lb.calculateDensity()
        lb.calculateVelocity()

        # apply collision
        lb.collision()

        # plotting for shear wave decay - density
        # densities are equal along y axis so simply take one value for each x for visualization
        if DENSITY: 
            q.append(np.max(np.abs(lb.rho - rhoInitial)))
            if i % SAVE_EVERY == 0:
                if plotting:
                    lb.updateDensityFigure(lb.rho)
                data = lb.rho[Ny//2,:]
                # amplitude = round(np.max(data) - np.min(data), 2)
                if plotting: 
                    densitySWFig, densitySWAx = plt.subplots()
                    densitySWAx.set_xlabel('x')
                    densitySWAx.set_ylabel('density')
                    densitySWAx.plot(range(Nx), data)
                    densitySWAx.set_ylim([rhoInitial-epsilon,rhoInitial+epsilon])
                    densitySWAx.set_title(f'timestep: {i}, w={omega}')
                    densitySWFig.savefig(f'./plots/shearwave/density_timestep{i}.png')
                    plt.close(densitySWFig)

        if VELOCITY:
            q.append(np.max(np.abs(lb.uy)))
            if i % SAVE_EVERY == 0:
                data = lb.uy[:,Nx//2]
                if plotting:
                    velocitySWFig, velocitySWAx = plt.subplots()
                    velocitySWAx.set_xlabel('y')
                    velocitySWAx.set_ylabel('velocity')
                    velocitySWAx.plot(range(Ny), data)
                    velocitySWAx.set_ylim([-epsilon,epsilon])
                    velocitySWAx.set_title(f'timestep: {i}, w={omega}')
                    velocitySWFig.savefig(f'./plots/shearwave/velocity_timestep{i}.png')
                    plt.close(velocitySWFig)
    return q

def shearwave(): 
    lb = LB()
    lb.omega = omega 
    lb.Nx = Nx
    lb.Ny = Ny 
    F = []
    
    if DENSITY:
        F = np.zeros((Ny,Nx,9))
        rho0 = rhoInitial
        for x in range(Nx):
            F[:,x,0] = rho0 + epsilon * np.sin(2*np.pi/Nx*x)

    if VELOCITY:
        F = np.zeros((Ny,Nx,9))
        F[:,:,0] = np.ones((Ny,Nx))
        for y in range(Ny):
            F[y,:,1] = epsilon * np.sin(2*np.pi/Ny*y)

    lb.__init__(F)

    # one simulation with the above parameters 
    simulate(lb, timesteps)
   
    # simulate different omegas and plot them
    aV = []
    mV = []
    omegas = []
    omegaSteps = np.linspace(0.5, 1.4, 10)
    for i in omegaSteps:
        omegas.append(i)
        lb.omega = i
        F = []
        if DENSITY:
            F = np.zeros((Ny,Nx,9))
            rho0 = rhoInitial
            for x in range(Nx):
                F[:,x,0] = rho0 + epsilon * np.sin(2*np.pi/Nx*x)
        if VELOCITY:
            F = np.zeros((Ny,Nx,9))
            F[:,:,0] = np.ones((Ny,Nx))
            for y in range(Ny):
                F[y,:,1] = epsilon*np.sin(2*np.pi/Ny*y)
        
        lb.__init__(F)
        q = simulate(lb, timesteps, plotting=False)

        if DENSITY:
            q = np.array(q)
            x = argrelextrema(q, np.greater)[0]
            q = q[x]
        else:
            x = np.arange(timesteps)

        simulated_viscosity = curve_fit(decay_perturbation, xdata=x, ydata=q)[0][0]
        analyticalViscosity = (1/3)*((1/lb.omega) - 0.5) 
        aV.append(analyticalViscosity)
        mV.append(simulated_viscosity)



    # plots measured against analytical viscosity after simulation 
    fig, ax = plt.subplots()
    ax.scatter(omegas,aV, marker='x', s=10)
    ax.scatter(omegas,mV, marker='x', s=10)
    plt.title("Measured and analytical viscosity for different omegas \n orange = measured viscosity, blue = analytical viscosity")
    plt.xlabel("omega")
    plt.ylabel("viscosity")
    fig.savefig(f'plots/shearwave/viscosity.png')

    
        
    
if __name__ == '__main__':
    shearwave()