# Astrix

Astrix (AStrophysical fluid dynamics on TRIangular eXtreme grids) is a package for numerical fluid dynamics on an unstructured, triangular grid. It uses a multidimensional upwind residual distribution scheme with explicit Runge Kutta time integration. The current version does two spatial dimensions only. Astrix is designed to run on NVidia GPUs.   


## Installation

Make sure CUDA is installed and nvcc is in your PATH.

1. Obtain the code: `git clone https://github.com/SijmeJan/Astrix`
2. Do `cd Astrix && make`, which will build Astrix itself, a visualization program `visAstrix` and Doxygen documentation.

## Usage

Examples will be provided. 
