#!/usr/bin/python

import numpy as np

def CanVertexBeTranslatedX(a, N):
    """Check whether a vertex is a periodic x vertex.

    For a mesh that is periodic in x, some triangles 'wrap around' and therefore need to look at the other side of the mesh for the coordinates of one of their vertices. This function checks if this is the case for vertex a.

    :param a: Vertex input to consider.
    :param N: Total number of vertices in Mesh

    :type a: int
    :type N: int

    :returns: 1 if a can be found one period away towards positive x, -1 if a can be found one period away towards negative x, and zero otherwise.
    :rtype: int
    """
    # Can be translated to left?
    if (a >= 4*N or (a >= N and a < 2*N) or (a >= -2*N and a < -N)):
        return 1

    # Can be translated to right?
    if (a < -3*N or (a >= 2*N and a < 3*N) or (a >=-N and a < 0)):
        return -1

    # Neither left nor right
    return 0

def CanVertexBeTranslatedY(a, N):
    """Check whether a vertex is a periodic y vertex.

    For a mesh that is periodic in y, some triangles 'wrap around' and therefore need to look at the other side of the mesh for the coordinates of one of their vertices. This function checks if this is the case for vertex a.

    :param a: Vertex input to consider.
    :param N: Total number of vertices in Mesh

    :type a: int
    :type N: int

    :returns: 1 if a can be found one period away towards positive y, -1 if a can be found one period away towards negative y, and zero otherwise.
    :rtype: int
    """
    # Can be translated to bottom?
    if (a < -N):
        return -1

    # Can be translated to top?
    if (a >= 2*N):
        return 1

    # Neither top nor bottom
    return 0

def GetCoordinates(a, vertX, vertY, Px, Py):
    """Get coordinates of vertex number a.

    For a mesh that is periodic in x, some triangles 'wrap around' and therefore need to look at the other side of the mesh for the coordinates of one of their vertices. This function gets the 'proper' coordinates of vertex a.

    :param a: Vertex input to consider.
    :param vertX: array of vertex x coordinates
    :param vertY: array of vertex y coordinates
    :param Px: Period in x
    :param Py: Period in y

    :type a: int
    :type vertX: ndarray
    :type vertY: ndarray
    :type Px: float
    :type Py: float

    :returns: x and y coordinate of vertex a.
    :rtype: float, float
    """
    nVertex = len(vertX)

    dxa = CanVertexBeTranslatedX(a, nVertex)*Px
    dya = CanVertexBeTranslatedY(a, nVertex)*Py

    while (a >= nVertex):
        a -= nVertex
    while (a < 0):
        a += nVertex

    return vertX[a] + dxa, vertY[a] + dya

def readVertex(read_direc, read_indx):
    """Reads in vertex data from Astrix simulation at specified time.

    Read the vertex output contained in directory read_direc at snapshot read_indx.

    :param read_direc: Directory containing Astrix output.
    :param read_indx: Output number to read.

    :type read_direc: string
    :type read_indx: int

    :returns: x-coordinates of the vertices; y-coordinates of the vertices
    :rtype: ndarray, ndarray
    """
    fname = 'vert%(#)04d.dat' % {"#": read_indx}
    f = open(read_direc + fname, "rb")
    data = np.fromfile(f, dtype = np.int32, count = 3, sep = "")

    sizeofData = data[1]
    nVertex = data[2]

    dt = np.float32

    if (sizeofData == 8):
         dt = np.float

    data = np.fromfile(f, dtype = dt, count = -1, sep = "")

    vertX = data[0 : nVertex]
    vertY = data[nVertex : 2*nVertex]

    return vertX, vertY

def readTriangle(read_direc, read_indx):
    """Reads in triangle data from Astrix simulation at specified time.

    Read the triangle output (i.e. the vertices belonging to every triangle) contained in directory read_direc at snapshot read_indx. Note that in case of periodic meshes, the entries may be smaller than zero or larger than the number of vertices.

    :param read_direc: Directory containing Astrix output.
    :param read_indx: Output number to read.

    :type read_direc: string
    :type read_indx: int

    :returns: a connectivity array of length 3 times the number of triangles. The first three entries represent the first triangle, etc.
    :rtype: ndarray
    """
    fname = 'tria%(#)04d.dat' % {"#": read_indx}
    data = np.fromfile(read_direc + fname,
                       dtype = np.int32, count = -1, sep = "")

    nTriangle = data[0]

    tv1 = data[1 : 1 + nTriangle]
    tv2 = data[1 + nTriangle : 1 + 2*nTriangle]
    tv3 = data[1 + 2*nTriangle: 1 + 3*nTriangle]

    conn = np.zeros(3*nTriangle)
    for n in range(0, nTriangle):
        conn[3*n + 0] = tv1[n]
        conn[3*n + 1] = tv2[n]
        conn[3*n + 2] = tv3[n]

    return conn

def readState(read_direc, read_indx, nVertex):
    """Reads in simulation data from Astrix simulation at specified time.

    Read the simulation output (i.e. the state at every vertex) contained in directory read_direc at snapshot read_indx.

    :param read_direc: Directory containing Astrix output.
    :param read_indx: Output number to read.
    :param nVertex: total number of vertices in Mesh

    :type read_direc: string
    :type read_indx: int
    :type nVertex: int

    :returns: four arrays containing density, x-velocity, y-velocity and total energy
    :rtype: ndarray, ndarray, ndarray, ndarray
    """
    fname = 'dens%(#)04d.dat' % {"#": read_indx}
    f = open(read_direc + fname, "rb")
    sizeofData = np.fromfile(f, dtype = np.int32, count = 1, sep = "")

    dt = np.float32
    if (sizeofData == 8):
         dt = np.float

    currTime = np.fromfile(f, dtype = dt, count = 1, sep = "")
    timeStep = np.fromfile(f, dtype = np.int32, count = 1, sep = "")

    data = np.fromfile(f, dtype = dt, count = -1, sep = "")
    dens = data[0 : nVertex]

    fname = 'momx%(#)04d.dat' % {"#": read_indx}
    f = open(read_direc + fname, "rb")
    sizeofData = np.fromfile(f, dtype = np.int32, count = 1, sep = "")

    dt = np.float32
    if (sizeofData == 8):
         dt = np.float

    currTime = np.fromfile(f, dtype = dt, count = 1, sep = "")
    timeStep = np.fromfile(f, dtype = np.int32, count = 1, sep = "")

    data = np.fromfile(f, dtype = dt, count = -1, sep = "")
    velx = data[0 : nVertex]/dens

    fname = 'momy%(#)04d.dat' % {"#": read_indx}
    f = open(read_direc + fname, "rb")
    sizeofData = np.fromfile(f, dtype = np.int32, count = 1, sep = "")

    dt = np.float32
    if (sizeofData == 8):
         dt = np.float

    currTime = np.fromfile(f, dtype = dt, count = 1, sep = "")
    timeStep = np.fromfile(f, dtype = np.int32, count = 1, sep = "")

    data = np.fromfile(f, dtype = dt, count = -1, sep = "")
    vely = data[0 : nVertex]/dens

    fname = 'ener%(#)04d.dat' % {"#": read_indx}
    f = open(read_direc + fname, "rb")
    sizeofData = np.fromfile(f, dtype = np.int32, count = 1, sep = "")

    dt = np.float32
    if (sizeofData == 8):
         dt = np.float

    currTime = np.fromfile(f, dtype = dt, count = 1, sep = "")
    timeStep = np.fromfile(f, dtype = np.int32, count = 1, sep = "")

    data = np.fromfile(f, dtype = dt, count = -1, sep = "")
    ener = data[0 : nVertex]

    return dens, velx, vely, ener

def readall(read_direc, read_indx, Px=1.0, Py=1.0):
    """Reads in data from Astrix simulation at specified time.

    Read the simulation output contained in directory read_direc at snapshot read_indx. If the mesh is periodic in x or y or both, the periods must be supplied as Px and Py so that we can create the full mesh.

    :param read_direc: Directory containing Astrix output.
    :param read_indx: Output number to read.
    :param Px: Optional distance in x over which the mesh is periodic.
    :param Py: Optional distance in y over which the mesh is periodic.

    :type read_direc: string
    :type read_indx: int
    :type Px: float
    :type Py: float

    :returns: Coordinates (x,y) of the vertices as a (Nv, 2) array, where Nv is the number of vertices; triangulation as a (Nt, 3) array, where Nt is the number of triangles; state as a (Nv, 4) array, containing density, two velocities and the total energy.
    :rtype: ndarray(Nv,2), ndarray(Nt,3), ndarray(Nv,4)
    """
    vertX, vertY = readVertex(read_direc, read_indx)
    nVertex = len(vertX)

    conn = readTriangle(read_direc, read_indx)
    dens, velx, vely, ener = readState(read_direc, read_indx, nVertex)

    connUniq = np.unique(conn)

    nPoints = len(connUniq)

    coords = np.zeros((nPoints, 2))
    state = np.zeros((nPoints, 4))

    for n in range(0, nPoints):
        a = connUniq[n]

        x, y = GetCoordinates(a, vertX, vertY, Px, Py)
        coords[n, 0] = x
        coords[n, 1] = y

        while (a >= nVertex):
            a -= nVertex
        while (a < 0):
            a += nVertex

        state[n, 0] = dens[a]
        state[n, 1] = velx[a]
        state[n, 2] = vely[a]
        state[n, 3] = ener[a]

    for n in range(0, len(conn)):
        conn[n] -= connUniq[0]
    for n in range(1, nPoints):
        connUniq[n] -= connUniq[0]
    connUniq[0] = 0

    index = np.zeros(connUniq[nPoints-1] + 1);
    for n in range(0, nPoints):
        index[connUniq[n]] = n

    newConn = np.zeros(len(conn))
    for n in range(0, len(conn)):
        newConn[n] = index[conn[n]];
    triang = newConn.reshape(len(conn)/3, 3)

    return coords, triang, state
