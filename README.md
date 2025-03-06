# PINN_3DSSE
This repository provides the python codes for Physics-Infomred Neural Networks to estimate the frictional parameters from geodetic observation.

# Paper information:
- Title: Physics-Informed Deep Learning for Estimating the Spatial Distribution of Frictional Parameters in Slow Slip Regions
- Authors: Rikuto Fukushima, Masayuki Kano, Kazuro Hirahara, Makiko Ohtani, Kyungjae Im, and Jean-Philippe Avouac
- Journal: Journal of Geophysical Research: Solid Earth
- Abstract: Slow slip events (SSEs) have been observed in many subduction zones and are understood to result from frictional unstable slip on the plate interface. The diversity of their characteristics and the fact that interplate slip can also be seismic suggest that frictional properties are heterogeneous. We are however lacking methods to constrain spatial distribution of frictional properties. In this paper, we employ Physics-Informed Neural Networks (PINNs) to achieve this goal using a synthetic model inspired by the long-term SSEs observed in the Bungo channel. PINN is a deep learning technique which can be used to solve the physics-based differential equations and determine the model parameters from observations. To examine the potential of our proposed method, we execute a series of numerical experiments. We start with an idealized case where it is assumed that fault slip is directly observed. We next move to a more realistic case where the synthetic surface displacement velocity data are observed by virtual GNSS stations. The geometry and friction properties of the velocity weakening region, where the slip instability develops, are well estimated, especially if surface displacement velocities above the velocity weakening region are observed. Our PINN-based method can be seen as an inversion technique with the regularization constraint that fault slip obeys a particular friction law. This approach remediates the issue that standard regularization techniques are based on non-physical constraints. These results of numerical experiments reveal that the PINN-based method is a promising approach for estimating the spatial distribution of friction parameters from GNSS observation.

# System requirements:
- Windows 11
- Python 3.10.9
- Pytorch 2.0.0

# Reproducing figures in the paper
I provide the result of PINN-based estimation from virtual 120 GNSS stations, which is mentioned in Section 3.2 of the paper.
You can make the various plots (learning curve, frictional parameters, fault slip velocity, crustal deformation, etc) with the Jupyter notebook file "Plot.ipynb" by loading the optimized neural network parameters.

# Frictional paramter estimation with PINN
"PINNs_for_3DSSE_RSFparam_Est.py" is the Python code to conduct the PINN-based frictional parameter inversion from GNSS observation. The optimized neural network parameters are saved as .pth file, and you can make the plot of the result by using "Plot.ipynb".

This code needs to load the following information from the other file:
1. (synthetic) observation data: surface displacement velocity data observed at each GNSS station
2. Green Function for static stress change K[i,j]: the matrix which shows how fault slip at cell j causes the static stress change to fault cell i
3. Green Function for surface deformation K_obs[i,j]: the matrix which shows how fault slip at cell j causes the deformation at GNSS station i
4. Initial condition:  slip velocity and state variable at each cell when t = 0
   
I provide the initial condition used in my numerical experiments as "InitialCondition.dat", but due to the large file size, I did not attach the synthetic observation data and Green Functions here. Note that these Green Functions can be made by using DC3D code (https://www.bosai.go.jp/e/dc3d.html).
