# Implementation of the Lattice Boltzmann method

`LB.py` in `src/parallelLB` contains the Lattice Boltzmann class which holds the relevant parameters such as the current state, omega, the grid size and the functions to simulate streaming and collision. `src/parallelLB/boundaries.py` contains the implementations of the boundary conditions. The code contains many explanatory comments.  
Analogously `src/seqLB` contains the sequential implementation. 

To run the simulations: 

 - Shear Wave Decay: `python /src/shearwave.py`
 - Couette Flow: `python /src/couetteflow.py`
 - Poiseuille Flow: `python /src/poiseuilleflow.py`
 - Sliding Lid: `python /src/slidinglid.py`
 - Sliding Lid with the parallel LB implementation (used to evaluate the efficiency of the parallelization): `mpiexec -n [number of workers] python .\src\slidinglidParallel.py`. Measures execution time and MLUPS. 

The parameters used in each simulation are defined at the top in the respective files and may be changed there. The default values for omega, Nx, Ny etc. are those used to create the plots shown in the report. The plots produced by each simulation can be found in `src/plots/{experiment name}`. 

### Example charts of the simulations 

#### Sliding lid 
||||
|:-:|:-:|:-:|
![](./src/plots/slidinglid/slidinglidimg_500.png) Step 500 | ![](./src/plots/slidinglid/slidinglidimg_2000.png) Step 2000 | ![](./src/plots/slidinglid/slidinglidimg_5000.png) Step 5000
![](./src/plots/slidinglid/slidinglidimg_10000.png) Step 10000 | ![](./src/plots/slidinglid/slidinglidimg_20000.png) Step 20000 | ![](./src/plots/slidinglid/slidinglidimg_50000.png) Step 50000

#### Shear wave density decay
||||
|:-:|:-:|:-:|
![](./src/plots/shearwave/density_timestep0.png) | ![](./src/plots/shearwave/density_timestep50.png) | ![](./src/plots/shearwave/density_timestep250.png)
![](./src/plots/shearwave/density_timestep550.png) | ![](./src/plots/shearwave/density_timestep1050.png) | ![](./src/plots/shearwave/density_timestep1750.png)

#### Couette flow
||||
|:-:|:-:|:-:|
![](./src/plots/couetteflow/img_0.png) | ![](./src/plots/couetteflow/img_500.png) | ![](./src/plots/couetteflow/img_1000.png)
![](./src/plots/couetteflow/img_1500.png) | ![](./src/plots/couetteflow/img_2000.png) | ![](./src/plots/couetteflow/img_2500.png)

#### Poiseuille flow
||||
|:-:|:-:|:-:|
![](./src/plots/poiseuilleflow/img_0.png) | ![](./src/plots/poiseuilleflow/img_1000.png) | ![](./src/plots/poiseuilleflow/img_2500.png)
![](./src/plots/poiseuilleflow/img_3000.png) | ![](./src/plots/poiseuilleflow/img_3500.png) | ![](./src/plots/poiseuilleflow/img_4000.png)

