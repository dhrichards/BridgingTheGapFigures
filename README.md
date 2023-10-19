# BridgingTheGapFigures
Code to reproduce results in "Bridging the gap between experimental and natural fabrics: Modelling ice stream fabric evolution and its comparison with ice core data"

# Requirements

The code requires the following data sets:

- Greenland velocity data from https://doi.org/10.5067/QUA5Q9SVMSJG
- Greenland surface height data from https://doi.org/10.5067/FPSU0V1MWUB6
- 2D EGRIP paths can be downloaded from the supplementary material to https://doi.org/10.5194/tc-15-3655-2021

EGRIP eigenvalue data is available from https://doi.org/10.1594/PANGAEA.949248 and pole figure data from https://doi.org/10.5281/zenodo.8015759

The code has the following dependencies

- numpy
- scipy
- matplotlib
- cartopy
- pyproj
- netCDF4
- pandas
- tqdm
- pickle
- specfab: https://github.com/nicholasmr/specfab



# Reproducing figures

2D surface paths can be created by running 'Save2DpathGerber.py', which tracks a 2D path upstream of EGRIP and extracts velocity, surface and bed data from the required data sets

The results of the paper can be reproduced by running 'plotdivide.py' and 'doubleegrip.py'



