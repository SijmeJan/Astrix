// -*-c++-*-
/*! \file vtk.cpp
\brief File containing functions to save legacy VTK files.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <cuda_runtime_api.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>

#include "../../Common/definitions.h"
#include "../../Mesh/mesh.h"
#include "./vtk.h"

namespace astrix {

//#########################################################################
/*! The constructor will determine if endian-swapping is necessary (VTK requires data to be written in bigendian form). Note that throughout we assume we are outputting 4 byte quantities (int's or float's). Upon return, \a swapEndian equals 1 when endian-swapping is necessary, 0 otherwise.*/
//#########################################################################

VTK::VTK()
{
  // Check if we need to swap endians
  swapEndian = 0;
  int tmp1 = 1;
  unsigned char *tmp2 = (unsigned char *) &tmp1;
  if (*tmp2 != 0)
    swapEndian = 1;
}

//#########################################################################
// Destructor; does nothing
//#########################################################################

VTK::~VTK()
{
}

//#########################################################################
/*! Write VTK output in file specified in \a fileName.

\param *fileName Name of output file
\param *mesh Pointer to Mesh object to output
\param *state Pointer to state vector. This is assumed to be density, velocities and pressure.
*/
//#########################################################################

void VTK::Write(const char *fileName, Mesh *mesh, realNeq *state)
{
  int nVertex = mesh->GetNVertex();
  int nTriangle = mesh->GetNTriangle();
  const int3 *triangleVertices = mesh->TriangleVerticesData();
  const real2 *vertexCoordinates = mesh->VertexCoordinatesData();
  real Px = mesh->GetPx();
  real Py = mesh->GetPy();

  // Fill connectivity array
  int *conn = new int[3*nTriangle];
  for (int i = 0; i < nTriangle; i++) {
    conn[3*i + 0] = triangleVertices[i].x;
    conn[3*i + 1] = triangleVertices[i].y;
    conn[3*i + 2] = triangleVertices[i].z;
  }

  // Complications arise for periodic meshes: extra points have to be added
  // where the mesh 'wraps around'

  // First, sort connectivity vector
  std::vector<int> sortConn(conn, conn + 3*nTriangle);
  std::sort(sortConn.begin(), sortConn.end());

  // Find unique values
  std::vector<int>::iterator it =
    std::unique(sortConn.begin(), sortConn.end());
  // Number of unique entries is the number of points required
  int nPoints = std::distance(sortConn.begin(),it);

  // Coordinates of vertices
  float *pts = new float[3*nPoints];

#if N_EQUATION == 4
  // Density, velocity, pressure
  int nVars = 3;
  int varDim[3] = {1, 3, 1};
  int centering[3] = {1, 1, 1};
  const char *varNames[3] = {"Density", "Velocity", "Pressure"};

  float *var1 = new float[nPoints];
  float *var2 = new float[3*nPoints];
  float *var3 = new float[nPoints];

  float *vars[3] = {var1, var2, var3};
#endif

#if N_EQUATION == 3
  // Density, velocity
  int nVars = 2;
  int varDim[3] = {1, 3};
  int centering[3] = {1, 1};
  const char *varNames[3] = {"Density", "Velocity"};

  float *var1 = new float[nPoints];
  float *var2 = new float[3*nPoints];

  float *vars[3] = {var1, var2};
#endif

#if N_EQUATION == 1
  // Output just scalar
  int nVars = 1;
  int varDim[1] = {1};
  int centering[1] = {1};
  const char *varNames[1] = {"Scalar"};

  float *var1 = new float[nPoints];

  float *vars[1] = {var1};
#endif

  for (int i = 0; i < nPoints; i++) {
    int a = sortConn[i];

    // Find the coordinates of this point
    real dxa = 0.0;

    // Can be translated to left?
    if (a >= 4*nVertex ||
        (a >= nVertex && a < 2*nVertex) ||
        (a >= -2*nVertex && a < -nVertex)) dxa = Px;

    // Can be translated to right?
    if (a < -3*nVertex ||
        (a >= 2*nVertex && a < 3*nVertex) ||
        (a >=-nVertex && a < 0)) dxa = -Px;

    real dya = 0.0;

    // Can be translated to bottom?
    if (a < -nVertex) dya = -Py;

    // Can be translated to top?
    if (a >= 2*nVertex) dya = Py;

    while (a >= nVertex) a -= nVertex;
    while (a < 0) a += nVertex;

    // Put in coordinates array
    pts[3*i + 0] = vertexCoordinates[a].x + dxa;
    pts[3*i + 1] = vertexCoordinates[a].y + dya;
    pts[3*i + 2] = 0.0;

    // Find the state vector at this point
    int j = sortConn[i];
    while (j >= nVertex) j -= nVertex;
    while (j < 0) j += nVertex;

#if N_EQUATION == 4
    var1[i] = state[j].x;
    var2[3*i + 0] = state[j].y;
    var2[3*i + 1] = state[j].z;
    var2[3*i + 2] = 0.0;
    var3[i] = state[j].w;
#endif
#if N_EQUATION == 3
    var1[i] = state[j].x;
    var2[3*i + 0] = state[j].y;
    var2[3*i + 1] = state[j].z;
    var2[3*i + 2] = 0.0;
#endif
#if N_EQUATION == 1
    var1[i] = state[j];
#endif
  }

  // Now adjust connectivity for added points

  // First, make sure we do not have any negative values
  for (int i = 0; i < 3*nTriangle; i++)
    conn[i] -= sortConn[0];
  for (int i = 1; i < nPoints; i++)
    sortConn[i] -= sortConn[0];
  sortConn[0] = 0;

  // Set an indexing array...
  int *index = new int[sortConn[nPoints-1] + 1];
  for (int i = 0; i < nPoints; i++)
    index[sortConn[i]] = i;

  // New connectivity array
  int *newConn = new int[3*nTriangle];
  for (int i = 0; i < 3*nTriangle; i++)
    newConn[i] = index[conn[i]];

  // Write to file
  writeData(fileName, nPoints,
            pts, nTriangle, newConn,
            nVars, varDim, centering,
            varNames, vars);

  delete[] index;
  delete[] conn;
  delete[] newConn;

  delete[] pts;
  delete[] var1;
#if N_EQUATION == 4
  delete[] var2;
  delete[] var3;
#endif
#if N_EQUATION == 3
  delete[] var2;
#endif
}

//#########################################################################
/*! Write single four-byte quantity (int or float) to output file, forcing it to be big endian.

\param val Value to output
*/
//#########################################################################

template <class T>
void VTK::writeSingle(T val)
{
  // Force output to be big endian
  if (swapEndian == 1) {
    unsigned char *bytes = (unsigned char *) &val;
    unsigned char tmp = bytes[0];
    bytes[0] = bytes[3];
    bytes[3] = tmp;
    tmp = bytes[1];
    bytes[1] = bytes[2];
    bytes[2] = tmp;
  }

  outFile.write(reinterpret_cast<char*>(&val), sizeof(T));
}

//#########################################################################
/*! Write state vector to VTK file.

\param nVars Number of variables to output
\param *varDim Pointer to array of size \a nVars containing the dimension of the variables (must be either 1 or 3)
\param *centering Pointer to array of size \a nVars containing 0 if the variable is cell centred and 1 if it is node centred
\param *varName Pointer to array of size \a nVars containing the names of the variables to write
\param **vars List of \a nVars pointers to the variable data
\param nPoints Total number of points
\param nCells Total number of cells*/
//#########################################################################

void VTK::writeState(int nVars, int *varDim, int *centering,
                     const char * const * varName, float **vars,
                     int nPoints, int nCells)
{
  int firstScalar = 0, firstVector = 0;
  int nScalars = 0, nVectors = 0;

  outFile << "CELL_DATA " << nCells << "\n";

  // First write primary scalars and vectors
  for (int i = 0; i < nVars; i++) {
    if (centering[i] == 0) {
      int writeFlag = 0;

      if (varDim[i] == 1) {
        if (firstScalar == 0) {
          writeFlag = 1;
          outFile << "SCALARS " << varName[i] << " float\n";
          outFile << "LOOKUP_TABLE default\n";
          firstScalar = 1;
        } else {
          nScalars++;
        }
      }
      if (varDim[i] == 3) {
        if (firstVector == 0) {
          writeFlag = 1;
          outFile << "VECTORS " << varName[i] << " float\n";
          firstVector = 1;
        } else {
          nVectors++;
        }
      }

      if (writeFlag == 1) {
        for (int j = 0; j < nCells*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }

  // Add rest of scalars
  firstScalar = 0;
  if (nScalars > 0) {
    for (int i = 0; i < nVars; i++) {
      int writeFlag = 0;
      if (centering[i] == 0) {
        if (varDim[i] == 1) {
          if (firstScalar == 0) {
            firstScalar = 1;
          } else {
            writeFlag = 1;
            outFile << varName[i] <<  " 1 " << nCells << " float\n";
          }
        }
      }

      if (writeFlag == 1) {
        for (int j = 0; j < nCells*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }

  // Add rest of vectors
  firstVector = 0;
  if (nVectors > 0) {
    outFile << "FIELD FieldData " << nVectors << "\n";

    for (int i = 0; i < nVars; i++) {
      int writeFlag = 0;
      if (centering[i] == 0) {
        if (varDim[i] == 3) {
          if (firstVector == 0) {
            firstVector = 1;
          } else {
            writeFlag = 1;
            outFile << varName[i] <<  " 3 " << nCells << " float\n";
          }
        }
      }

      if (writeFlag == 1) {
        for (int j = 0; j < nCells*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }

  // Now write node-centred variables
  outFile << "POINT_DATA " << nPoints << "\n";

  firstScalar = 0;
  firstVector = 0;
  nScalars = 0;
  nVectors = 0;

  // First write primary scalars and vectors
  for (int i = 0; i < nVars; i++) {
    if (centering[i] != 0) {
      int writeFlag = 0;

      if (varDim[i] == 1) {
        if (firstScalar == 0) {
          writeFlag = 1;
          outFile << "SCALARS " << varName[i] << " float\n";
          outFile << "LOOKUP_TABLE default\n";

          firstScalar = 1;
        } else {
          nScalars++;
        }
      }
      if (varDim[i] == 3) {
        if (firstVector == 0) {
          writeFlag = 1;
          outFile << "VECTORS " << varName[i] << " float\n";

          firstVector = 1;
        } else {
          nVectors++;
        }
      }

      if (writeFlag == 1) {
        for (int j = 0; j < nPoints*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }

  // Write rest of scalars and vectors
  firstScalar = 0;
  if (nScalars > 0) {
    outFile << "FIELD FieldData " << nScalars << "\n";

    for (int i = 0; i < nVars; i++) {
      int writeFlag = 0;
      if (centering[i] != 0) {
        if (varDim[i] == 1) {
          if (firstScalar == 0) {
            firstScalar = 1;
          } else {
            writeFlag = 1;
            outFile << varName[i] <<  " 1 " << nPoints << " float\n";
          }
        }
      }

      if (writeFlag == 1) {
        for (int j = 0; j < nPoints*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }

  firstVector = 0;
  if (nVectors > 0) {
    outFile << "FIELD FieldData " << nVectors << "\n";

    for (int i = 0; i < nVars; i++) {
      int writeFlag = 0;
      if (centering[i] != 0) {
        if (varDim[i] == 3) {
          if (firstVector == 0) {
            firstVector = 1;
          } else {
            writeFlag = 1;
            outFile << varName[i] << " 3 " << nPoints << " float\n";
          }
        }
      }

      if (writeFlag == 1) {
        for (int j = 0; j < nPoints*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }
}

//#########################################################################
/*! All preparations done, actually write the VTK file.

\param *fileName Name of the file to write
\param nPoints Number of points to write
\param *pts Pointer to point coordinates, encoded as [x1, y1, z1, x2, ...]
\param nCells Number of cells (triangles)
\param *conn Connectivity array
\param nVars Number of state variables to write
\param *varDim Array of length \a nVars, specifying the dimension of each variable. Can be either 1 (scalar) or 3 (vector)
\param *centering Array of length \a nVars, specifying whether the variable is triangle-based (0) or point-based (1)
\param *varNames Array of length \a nVars containing the names of the variables
\param **vars Array of length \a nVars containing pointers to the data to write. Note that the size of vars[i] should be \a nPoints*\a varDim[i]
*/
//#########################################################################

void VTK::writeData(const char *fileName, int nPoints, float *pts,
                    int nCells, int *conn,
                    int nVars, int *varDim, int *centering,
                    const char * const *varNames, float **vars)
{
  // Open output file
  outFile.open(fileName);

  // Write header
  outFile << "# vtk DataFile Version 2.0\n";
  outFile << "Written using Astrix\n";
  outFile << "BINARY\n";
  outFile << "DATASET UNSTRUCTURED_GRID\n";
  outFile << "POINTS " << nPoints << " float\n";

  // Write vertex coordinates
  for (int i = 0; i < 3*nPoints; i++)
    writeSingle(pts[i]);

  int connSize = 4*nCells;
  outFile << "CELLS " << nCells << " " << connSize << "\n";

  // Write connectivity
  for (int i = 0; i < nCells; i++) {
    writeSingle(3);  // Only using triangles
    for (int j = 0; j < 3; j++)
      writeSingle(conn[3*i + j]);
  }

  // Only using triangles
  outFile << "CELL_TYPES " << nCells << "\n";
  for (int i = 0; i < nCells; i++)
    writeSingle(5);

  // Write state variables
  writeState(nVars, varDim, centering, varNames, vars, nPoints, nCells);

  // Close file
  outFile.close();
}

}  // namespace astrix
