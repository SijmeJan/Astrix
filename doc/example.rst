Running Astrix
=========================

Command line options
-------------------------------

Issueing ``astrix`` gives::

    Usage: astrix [-d] [-v verboseLevel] [-D debugLevel] [-r restartNumber] [-cl conservationLaw] filename
    -d                  : run on GPU device
    -v verboseLevel     : amount of output to stdout (0 - 2)
    -D debugLevel       : amount of extra checks for debugging
    -r restartNumber    : try to restart from previous dump
    -cl conservationLaw : use different conservation law. Can be
                          either "advect", "burgers"
                          "cart_iso" or "cart_euler"
    filename            : input file name

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
* ``source/`` : Rayleigh-Taylor instability.
* ``vortex/`` : Isentropic vortex test.

In addition, some scalar equation tests can be found in
``Astrix/run/scalar``. A suite of test problems can be run by
entering, in the ``Astrix`` directory::
  python python/astrix/testsuite.py ./
which will generate a pdf document with outputs from most test
problems.
