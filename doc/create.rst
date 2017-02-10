Creating a new setup
=========================

Going beyond the test problems supplied requires a bit of
coding. It is instructive to look at places in the code that depend on
the ``ProblemDefinition`` specifying which test is being run. For
example, in the ``Astrix`` directory enter::

  grep -r PROBLEM_CYL src/astrix/*

to see a list of places where code specific to the test problem of the
flow around a cylinder is executed. These will appear as part of the
Mesh class, setting up the initial mesh with the inner hole, as well
as in the Simulation class, most notably to set initial and boundary
conditions. Additional setups can be added by adding an entry to the
``ProblemDefinition`` enum in ``definitions.h``, and create specific
mesh generation instructions, initial and boundary conditions for this
new setup.
