Running Astrix
=========================

Command line options
-------------------------------

.. program-output:: ../bin/astrix

Test problems
-------------------------------

A few test problems for the Euler equations are supplied in the ``Astrix/run/euler`` directory. Each directory contains an input file ``astrix.in`` specifying parameters of the simulation. See the classes :ref:`label-mesh-parameter` and :ref:`label-simulation-parameter` for details. Note that all parameters have to be present in the input file. For each test problem, run Astrix in the respective directory:


* ``blast/`` : A one-dimensional problem of two interacting blast waves.
* ``cyl/`` : supersonic flow around a cylinder.
* ``kh/`` : Kelvin-Helmholtz instability.
* ``linear/`` : A one-dimensional problem of a linear sound wave.
* ``noh/`` : The Noh test problem.
* ``riemann/`` : Two-dimensional Riemann problem.
* ``sod/`` : A one-dimensional shock tube.
* ``vortex/`` : Isentropic vortex test.
