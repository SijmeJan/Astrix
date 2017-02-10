Output and Visualisation
=========================

Simulation output comes in three kinds:

* Raw data of both Mesh and Simulation. Every save interval, both the Mesh and the state are written to disc. Mesh information is written in three files: ``vert####.dat``, containing vertex coordinates, ``tria####.dat``, containing triangle information (vertices and edges), and ``edge####.dat``, containing edge information (triangles). Here and in the following, ``####`` stands for a four-digit number, e.g. ``0001``, ``0199``. The state vector is written in four files ``dens####.dat``, containing the density, ``momx####.dat`` containing the x-momentum, ``momy####.dat`` containing the y-momentum and ``ener####.dat`` containing the total energy.
* Fine grain global data in ``simulation.dat``. Every fine grain save interval, a new line is added to this ASCII file, containing global simulation quantities (simulation time plus other quantities that might be interesting to monitor).
* When desired, Astrix can output legacy VTK files for easy visualisation for example with the open source package VisIt (available from https://wci.llnl.gov/simulation/computer-codes/visit)

All raw data files are binary files. The format for each is as follows:

* ``vert####.dat``: three 4 byte integers specifying the number of spatial dimensions (fixed to two for now), the size of each element (can be 4 byte float or 8 byte double), and the total number of vertices Nv. These are followed by the actual data: Nv x coordinates followed by Nv y coordinates.
* ``tria####.dat``: a 4 byte integer specifying the total number of triangles Nt. This is followed by Nt integers specifying the first vertex for each triangle, followed by Nt integers specifying the second vertex for each triangle, followed by Nt integers specifying the third vertex for each triangle. After this, the edges for every triangle: three sets of Nt integers specifying the first, second and third edge for every triangle.
* ``edge####.dat``: a 4 byte integer specifying the total number of edges Ne. This is followed by Ne integers specifying the first neighbouring triangle, followed by Ne integers specifying the second neighbouring triangle. These entries can be -1 if the edge happens to have only one neighbouring triangle.
* ``dens####.dat`` (and similar for the other state vector files): a 4 byte integer specifying the size of the output (can be float or double), a float or double specifying the current simulation time, followed by an 4 byte integer specifying the number of time steps taken so far. Then the actual data follows for each vertex a double or a float.

Constructing periodic meshes from the raw outputs can be tricky. A python script to read raw data files into a format that can be used for example with matplotlib is provided in the ``Astrix/python/astrix/`` directory. The module ``readfiles`` contains a function ``readall`` that reads in both Mesh and Simulation data. Provided ``Astrix/python/`` is in your ``PYTHONPATH`` environmental variable, this function can be used for plotting as follows::

                import numpy as np
                import matplotlib.pyplot as plt
                import astrix.readfiles as a

                # Read in data
                coords, triang, state = a.readall("path/to/data/", 0)

                # Vertex coordinates
                x = coords[:, 0]
                y = coords[:, 1]

                # Density
                d = state[:, 0]

                # Contour levels
                levels = np.linspace(np.min(d), np.max(d), 100)

                fig = plt.figure(figsize=(5.5, 5.5))
                ax = fig.add_subplot(1, 1, 1)

                # Plot triangulation
                ax.triplot(x, y, triang, 'k-')
                # Contour plot density
                ax.tricontourf(x, y, triang, d, levels=levels, cmap=cm.bwr)

                plt.show()

The complete content of the ``astrix.readfiles`` module is given below. In most cases, the ``readall`` function is all that is required.

.. automodule:: astrix.readfiles
                :members:
