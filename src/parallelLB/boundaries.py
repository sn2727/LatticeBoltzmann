import numpy as np 

"""
Implementation of the different boundaries
if boundaries are used in simulation, their before() function is called before streaming()
there we usually cache some values
The after() method is called after streaming and collision to adjust F according to the boundary
"""


# maps velocity vectors to their opposite velocity vector with respect to ux,uy
OPPOSITE_IDXS = np.ascontiguousarray([0,5,6,7,8,1,2,3,4])

class RightWallBoundary():
    lb = None 
    # indexes of the velocity vectors pointing to the right (x component = 1)
    idxs = [2,3,4]
    tmp = []

    def __init__(self, lb):
        self.lb = lb 
        self.tmp = lb.F[:, -2]

    def before(self):
        self.tmp = self.lb.F[:, -2]

    def after(self):
        self.lb.F[:, -1, self.idxs] = self.tmp[:, OPPOSITE_IDXS[self.idxs]]
        self.lb.ux[:,-1] = 0.0

class LeftWallBoundary():
    # indexes of the velocity vectors pointing to the left (x component = -1)
    idxs = [6,7,8]
    tmp = []
    lb = None 

    def __init__(self, lb):
        self.lb = lb
        self.tmp = lb.F[:, 1]

    def before(self):
        self.tmp = self.lb.F[:, 1]

    def after(self):
        self.lb.F[:, 0, self.idxs] = self.tmp[:, OPPOSITE_IDXS[self.idxs]]
        self.lb.ux[:,0] = 0.0

class BottomWallBoundary():
    # indexes of the velocity vectors pointing to the bottom (y component = -1)
    idxs = [1,2,8]
    tmp = []
    lb = None  

    def __init__(self, lb): 
        self.lb = lb
        self.tmp = lb.F[1]  

    def before(self):
        self.tmp = self.lb.F[1]
    
    def after(self):
        self.lb.F[0, :, self.idxs] = (self.tmp[:, OPPOSITE_IDXS[self.idxs]]).T
        self.lb.ux[0,:] = 0.0
        self.lb.uy[0,:] = 0.0


class TopWallBoundary():
    lb = None 
    # indexes of the velocity vectors pointing to the top (y component = 1)
    idxs = [4, 5, 6]
    tmp = []

    def __init__(self, lb):
        self.lb = lb
        self.tmp = lb.F[-2]

    def before(self):
        # cache values of second to last row regarding top wall 
        self.tmp = self.lb.F[-2]

    def after(self):
        # OPPOSITE_IDXS returns the opposite direction vectors 
        # set stored values for the velocity vectors pointing down to stored oppsite velocity values 
        self.lb.F[-1, :, self.idxs] = self.tmp[:, OPPOSITE_IDXS[self.idxs]].T
        # set velocities of particles at wall to 0
        self.lb.ux[-1,:] = 0.0
        self.lb.uy[-1,:] = 0.0
        

class MovingBottomWallBoundary():
    lb = None 
    idxs = [1,2,8]
    wallVelocity = 0
    cs = 1/np.sqrt(3)
    tmp = []

    def __init__(self, lb, wallVelocity):
        self.lb = lb 
        self.tmp = lb.F[1]
        self.wallVelocity = wallVelocity

    # this function is supposed to be called before streaming to store the state before the streaming  
    def before(self):
        # cache second to last row at bottom  
        self.tmp = self.lb.F[1]
    
    # this function is supposed to be called after streaming to account for the boundary 
    # by using of the temporarilty stored value in before()
    def after(self):
        # apply momentum 
        # velocity vectors redefined here as C in one vector for convenience 
        C = np.ascontiguousarray(
             np.array([[0,1,1,0,-1,-1,-1,0,1], 
                       [0,0,1,1,1,0,-1,-1,-1]]).T
        )

        density = np.sum(self.lb.F[1], axis=-1)
        multiplier = 2 * (1/self.cs) ** 2
        momentum = multiplier * (C @ [0.0, self.wallVelocity]) * (self.lb.weights * density[:, None])
        momentum = momentum[:, OPPOSITE_IDXS[self.idxs]]
        self.lb.F[0, :, self.idxs] = (self.tmp[:, OPPOSITE_IDXS[self.idxs]] - momentum).T
        self.lb.ux[0,:] = self.wallVelocity
        self.lb.uy[0,:] = self.wallVelocity


class MovingTopWallBoundary():
    # analog to MovingBottomWallBoundary
    lb = None 
    idxs = [4, 5, 6]
    wallVelocity = 0
    cs = 1/np.sqrt(3)
    tmp = []

    def __init__(self, lb, wallVelocity):
        self.lb = lb 
        self.tmp = lb.F[-2]
        self.wallVelocity = wallVelocity

    def before(self):
        self.tmp = self.lb.F[-2]
    
    def after(self):
        C = np.ascontiguousarray(
             np.array([[0,1,1,0,-1,-1,-1,0,1], 
                       [0,0,1,1,1,0,-1,-1,-1]]).T
        )

        density = np.sum(self.lb.F[-2], axis=-1)
        multiplier = 2 * (1/self.cs) ** 2
        momentum = multiplier * (C @ [0.0, self.wallVelocity]) * (self.lb.weights * density[:, None])
        momentum = momentum[:, OPPOSITE_IDXS[self.idxs]]
        self.lb.F[-1, :, self.idxs] = (self.tmp[:, OPPOSITE_IDXS[self.idxs]] - momentum).T
        self.lb.ux[-1,:] = self.wallVelocity
        self.lb.uy[-1,:] = self.wallVelocity
        
        

class HorizontalInletOutletBoundary():

    lb = None 
    inCache = None 
    outCache = None 

    def __init__(self, lb, n, pressure_in=0.201, pressure_out=0.2, cs=1/np.sqrt(3)):
        self.lb = lb 
        self.pIn = pressure_in / cs**2
        self.pOut = pressure_out / cs**2
        self.cs = cs
        self.n = n

    def before(self):
        # normal equilibrium 
        Feq = self.lb.calcFeq(self.lb.rho, self.lb.ux, self.lb.uy)

        # equilibrium for the inlet using pIn 
        inFeq = self.lb.calcFeq(
                self.pIn * self.lb.rho,
                self.lb.ux, self.lb.uy)
        
        # equilibrium for the outlet using pOut
        outFeq = self.lb.calcFeq(
                self.pOut * self.lb.rho,
                self.lb.ux, self.lb.uy)
        
        # only consider second and second to last column
        inFeq = inFeq[:, -2]
        outFeq = outFeq[:, 1]

        self.inCache = inFeq + (self.lb.F[:,-2] - Feq[:, -2])
        self.outCache = outFeq + (self.lb.F[:, 1] - Feq[:, 1])

    def after(self):
        # 2,3,4 are indexes of the right pointing velocity vectors
        self.lb.F[:, 0, [2,3,4]] = self.inCache[:, [2,3,4]]
        # 6,7,8 are indexes of the left pointing velocity vectors
        self.lb.F[:, -1, [6,7,8]] = self.outCache[:, [6,7,8]]