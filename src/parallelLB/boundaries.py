import numpy as np 

# maps velocity vectors to their opposite with respect to ux,uy
OPPOSITE_IDXS = np.ascontiguousarray([0,5,6,7,8,1,2,3,4])

class RightWallBoundary():
    lb = None 
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

class TopWallBoundary():
    lb = None  
    idxs = [1,2,8]
    tmp = []

    def __init__(self, lb): 
        self.lb = lb
        self.tmp = lb.F[1]  

    def before(self):
        self.tmp = self.lb.F[1]
    
    def after(self):
        self.lb.F[0, :, self.idxs] = (self.tmp[:, OPPOSITE_IDXS[self.idxs]]).T
        self.lb.ux[0,:] = 0.0
        self.lb.uy[0,:] = 0.0


class BottomWallBoundary():
    # indexes of vectors pointing downwards (y = -1)
    lb = None 
    idxs = [4, 5, 6]
    tmp = []

    def __init__(self, lb):
        self.lb = lb
        self.tmp = lb.F[-2]

    def before(self):
        # store values of second to last row regarding y coordinate
        self.tmp = self.lb.F[-2]

    def after(self):
        # set stored values for the velocity vectors pointing down to stored oppsite velocity values 
        self.lb.F[-1, :, self.idxs] = self.tmp[:, OPPOSITE_IDXS[self.idxs]].T
        # set velocities of points at bottom bounding to 0 
        self.lb.ux[-1,:] = 0.0
        self.lb.uy[-1,:] = 0.0
        

class MovingTopWallBoundary():
    lb = None 
    idxs = [1,2,8]
    wallVelocity = 0
    cs = 1/np.sqrt(3)
    tmp = []

    def __init__(self, lb, wallVelocity):
        self.lb = lb 
        self.tmp = lb.F[1]
        self.wallVelocity = wallVelocity

    def before(self):
        self.tmp = self.lb.F[1]
    
    def after(self):

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
        
        

class HorizontalInletOutletBoundary():

    lb = None 
    inlet_cache = None 
    outlet_cache = None 

    def __init__(self, lb, n, pressure_in=0.201, pressure_out=0.2, cs=1/np.sqrt(3)):
        self.lb = lb 
        self.p_in = pressure_in / cs**2
        self.p_out = pressure_out / cs**2
        self.cs = cs
        self.n = n

    def before(self):
        Feq = self.lb.calcFeq(self.lb.rho, self.lb.ux, self.lb.uy)

        inlet_feq = self.lb.calcFeq(
                self.p_in * np.ones((self.lb.Ny, self.lb.Nx)),
                self.lb.ux[:, -2], self.lb.uy[:, -2])
        

        outlet_feq = self.lb.calcFeq(
                self.p_out * np.ones((self.lb.Ny, self.lb.Nx)),
                self.lb.ux[:, 1], self.lb.uy[:, 1])

        self.inlet_cache = inlet_feq + (self.lb.F[:, -2] - Feq[:, -2])
        self.outlet_cache = outlet_feq + (self.lb.F[:, 1] - Feq[:, 1])

    def after(self):
        # print(f"Shape inlet cache {self.inlet_cache.shape}")
        # print(f"Shape self.inlet_cache[:, [6,7,8]] {self.inlet_cache[:, [6,7,8]].shape}")
        # print(f"Shape F[:, 0, [6,7,8]] {self.lb.F[:, 0, [6,7,8]].shape}")
        # print(f"Shape F {self.lb.F.shape}")
        self.lb.F[:, 0, [6,7,8]] = self.inlet_cache[:, 0, [6,7,8]]
        self.lb.F[:, -1, [2,3,4]] = self.outlet_cache[:, -1, [2,3,4]]
