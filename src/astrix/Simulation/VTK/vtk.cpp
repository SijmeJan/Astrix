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

VTK::VTK()
{
  // Check if we need to swap endians
  swapEndian = 0;
  int tmp1 = 1;
  unsigned char *tmp2 = (unsigned char *) &tmp1;
  if (*tmp2 != 0)
    swapEndian = 1;
}

VTK::~VTK()
{
}

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

  // Write the mesh
  write_unstructured_mesh(fileName, nPoints,
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
}

/* ****************************************************************************
 *  Function: force_big_endian
 *
 *  Purpose:
 *      Determines if the machine is little-endian.  If so, then, for binary
 *      data, it will force the data to be big-endian.
 *
 *  Note:       This assumes that all inputs are 4 bytes long.
 *
 *  Programmer: Hank Childs
 *  Creation:   September 3, 2004
 *
 * ************************************************************************* */

void VTK::ForceBigEndian(unsigned char *bytes)
{
  if (swapEndian == 1) {
    unsigned char tmp = bytes[0];
    bytes[0] = bytes[3];
    bytes[3] = tmp;
    tmp = bytes[1];
    bytes[1] = bytes[2];
    bytes[2] = tmp;
  }
}

template <class T>
void VTK::writeSingle(T val)
{
  ForceBigEndian((unsigned char *) &val);
  outFile.write(reinterpret_cast<char*>(&val), sizeof(T));
}

/* ****************************************************************************
 *  Function: write_variables
 *
 *  Purpose:
 *      Writes the variables to the file.  This can be a bit tricky.  The
 *      cell data must be written first, followed by the point data.  When
 *      writing the [point|cell] data, one variable must be declared the
 *      primary scalar and another the primary vector (provided scalars
 *      or vectors exist).  The rest of the arrays are added through the
 *      "field data" mechanism.  Field data should support groups of arrays
 *      with different numbers of components (ie a scalar and a vector), but
 *      there is a failure with the VTK reader.  So the scalars are all written
 *      one group of field data and then the vectors as another.  If you don't
 *      write it this way, the vectors do not show up.
 *
 *  Programmer: Hank Childs
 *  Creation:   September 3, 2004
 *
 * ************************************************************************* */

void VTK::write_variables(int nVars, int *varDim, int *centering,
                          const char * const * varName, float **vars,
                          int nPoints, int nCells)
{
  int first_scalar = 0, first_vector = 0;
  int num_scalars = 0, num_vectors = 0;

  outFile << "CELL_DATA " << nCells << "\n";

  /* The field data is where the non-primary scalars and vectors are
   * stored.  They must all be grouped together at the end of the point
   * data.  So write out the primary scalars and vectors first.
   */
  for (int i = 0; i < nVars; i++) {
    if (centering[i] == 0) {
      int should_write = 0;

      if (varDim[i] == 1) {
        if (first_scalar == 0) {
          should_write = 1;
          outFile << "SCALARS " << varName[i] << " float\n";
          outFile << "LOOKUP_TABLE default\n";
          first_scalar = 1;
        } else {
          num_scalars++;
        }
      }
      if (varDim[i] == 3) {
        if (first_vector == 0) {
          should_write = 1;
          outFile << "VECTORS " << varName[i] << " float\n";
          first_vector = 1;
        } else {
          num_vectors++;
        }
      }

      if (should_write) {
        for (int j = 0; j < nCells*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }

  first_scalar = 0;
  if (num_scalars > 0) {
    for (int i = 0; i < nVars; i++) {
      int should_write = 0;
      if (centering[i] == 0) {
        if (varDim[i] == 1) {
          if (first_scalar == 0) {
            first_scalar = 1;
          } else {
            should_write = 1;
            outFile << varName[i] <<  " 1 " << nCells << " float\n";
          }
        }
      }

      if (should_write) {
        for (int j = 0; j < nCells*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }

  first_vector = 0;
  if (num_vectors > 0) {
    outFile << "FIELD FieldData " << num_vectors << "\n";

    for (int i = 0; i < nVars; i++) {
      int should_write = 0;
      if (centering[i] == 0) {
        if (varDim[i] == 3) {
          if (first_vector == 0) {
            first_vector = 1;
          } else {
            should_write = 1;
            outFile << varName[i] <<  " 3 " << nCells << " float\n";
          }
        }
      }

      if (should_write) {
        for (int j = 0; j < nCells*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }

  outFile << "POINT_DATA " << nPoints << "\n";

  first_scalar = 0;
  first_vector = 0;
  num_scalars = 0;
  num_vectors = 0;
  /* The field data is where the non-primary scalars and vectors are
   * stored.  They must all be grouped together at the end of the point
   * data.  So write out the primary scalars and vectors first.
   */
  for (int i = 0; i < nVars; i++) {
    if (centering[i] != 0) {
      int should_write = 0;

      if (varDim[i] == 1) {
        if (first_scalar == 0) {
          should_write = 1;
          outFile << "SCALARS " << varName[i] << " float\n";
          outFile << "LOOKUP_TABLE default\n";

          first_scalar = 1;
        } else {
          num_scalars++;
        }
      }
      if (varDim[i] == 3) {
        if (first_vector == 0) {
          should_write = 1;
          outFile << "VECTORS " << varName[i] << " float\n";

          first_vector = 1;
        } else {
          num_vectors++;
        }
      }

      if (should_write) {
        for (int j = 0; j < nPoints*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }

  first_scalar = 0;
  if (num_scalars > 0) {
    outFile << "FIELD FieldData " << num_scalars << "\n";

    for (int i = 0; i < nVars; i++) {
      int should_write = 0;
      if (centering[i] != 0) {
        if (varDim[i] == 1) {
          if (first_scalar == 0) {
            first_scalar = 1;
          } else {
            should_write = 1;
            outFile << varName[i] <<  " 1 " << nPoints << " float\n";
          }
        }
      }

      if (should_write) {
        for (int j = 0; j < nPoints*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }

  first_vector = 0;
  if (num_vectors > 0) {
    outFile << "FIELD FieldData " << num_vectors << "\n";

    for (int i = 0; i < nVars; i++) {
      int should_write = 0;
      if (centering[i] != 0) {
        if (varDim[i] == 3) {
          if (first_vector == 0) {
            first_vector = 1;
          } else {
            should_write = 1;
            outFile << varName[i] << " 3 " << nPoints << " float\n";
          }
        }
      }

      if (should_write) {
        for (int j = 0; j < nPoints*varDim[i]; j++)
          writeSingle(vars[i][j]);
      }
    }
  }
}

/* ****************************************************************************
//  Function: write_unstructured_mesh
//
//  Purpose:
//      Writes out a unstructured mesh.
//
//
//  Arguments:
//      filename   The name of the file to write.  If the extension ".vtk" is
//                 not present, it will be added.
//      useBinary  '0' to write ASCII, !0 to write binary
//      nPoints       The number of points in the mesh.
//      pts        The spatial locations of the points.  This array should
//                 be size 3*nPoints.  The points should be encoded as:
//                 <x1, y1, z1, x2, y2, z2, ..., xn, yn, zn>
//      nCells     The number of cells.
//      celltypes  The type of each cell.
//      conn       The connectivity array.
//      nVars      The number of variables.
//      varDim     The dimension of each variable.  The size of varDim should
//                 be nVars.  If var i is a scalar, then varDim[i] = 1.
//                 If var i is a vector, then varDim[i] = 3.
//      centering  The centering of each variable.  The size of centering
//                 should be nVars.  If centering[i] == 0, then the variable
//                 is cell-based.  If centering[i] != 0, then the variable
//                 is point-based.
//      vars       An array of variables.  The size of vars should be nVars.
//                 The size of vars[i] should be nPoints*varDim[i].
//
//  Programmer: Hank Childs
//  Creation:   September 2, 2004
//
// ***************************************************************************/

void VTK::write_unstructured_mesh(const char *fileName, int nPoints, float *pts,
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

  for (int i = 0; i < 3*nPoints; i++)
    writeSingle(pts[i]);

  int conn_size = 4*nCells;
  outFile << "CELLS " << nCells << " " << conn_size << "\n";

  for (int i = 0; i < nCells; i++) {
    writeSingle(3);  // Only using triangles
    for (int j = 0; j < 3; j++)
      writeSingle(conn[3*i + j]);
  }

  // Only using triangles
  outFile << "CELL_TYPES " << nCells << "\n";
  for (int i = 0; i < nCells; i++)
    writeSingle(5);

  write_variables(nVars, varDim, centering, varNames, vars, nPoints, nCells);

  // Close file
  outFile.close();
}

}  // namespace astrix
