Astrix
========================

|build-status| |docs|

Astrix (AStrophysical fluid dynamics on TRIangular eXtreme grids) is a package for numerical fluid dynamics on an unstructured, triangular grid. It uses a multidimensional upwind residual distribution scheme with explicit Runge Kutta time integration. The current version does two spatial dimensions only. Astrix is designed to run on NVidia GPUs.


Quick start
------------------------------

Make sure CUDA is installed and nvcc is in your PATH.

1. Obtain the code: ``git clone https://github.com/SijmeJan/Astrix``
2. Do ``cd Astrix && make astrix``, which will build Astrix.

Now go into a directory of one of the test problems provided::

  cd run/euler/kh

and run Astrix::

  ../../../bin/astrix astrix.in

Output is generated (apart from raw data files) in the form of legacy
VTK files, which can be viewed for example with the free tool `Visit
<https://wci.llnl.gov/simulation/computer-codes/visit>`_

Documentation
-------------------------------

Full documentation is provided
http://astrix.readthedocs.io
