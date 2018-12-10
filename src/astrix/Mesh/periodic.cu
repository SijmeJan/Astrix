// -*-c++-*-
/*! \file periodic.cu
\brief Function for making initial mesh periodic

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./mesh.h"
#include "../Common/cudaLow.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"

namespace astrix {

//######################################################################
/*! \brief Check if edge between v1 and v2 lies in triangle i

\param i Triangle to be checked
\param v1 Vertex 1
\param v2 Vertex 2
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *eBetween Pointer to output*/
//######################################################################

__host__ __device__
void FindEdgeBetween(int i, unsigned int v1, unsigned int v2,
                     int3 *pTv, int3 *pTe, int *eBetween)
{
  if (pTv[i].y == (int) v1 && pTv[i].x == (int) v2) eBetween[0] = pTe[i].x;
  if (pTv[i].z == (int) v1 && pTv[i].y == (int) v2) eBetween[0] = pTe[i].y;
  if (pTv[i].x == (int) v1 && pTv[i].z == (int) v2) eBetween[0] = pTe[i].z;
}

//######################################################################
/*! \brief Kernel finding edge between v1 and v2

\param nTriangle Total number of triangles in Mesh
\param v1 Vertex 1
\param v2 Vertex 2
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEdgeBetween Pointer to output*/
//######################################################################

__global__ void
devFindEdgeBetween(int nTriangle, unsigned int v1, unsigned int v2,
                   int3 *pTv, int3 *pTe, int *pEdgeBetween)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    FindEdgeBetween(i, v1, v2, pTv, pTe, pEdgeBetween);

    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Once all boundary vertices are in place, we can make the initial mesh periodic if desired. This is done by adding extra triangles connecting for example the leftmost edge to the rightmost edge (in case the mesh should be periodic in x).*/
//######################################################################

void Mesh::MakePeriodic()
{
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();

  // Make periodic in x
  if (meshParameter->periodicFlagX) {
    Array<real> *vxTemp =
      new Array<real>(1, cudaFlag, (unsigned int) nVertex);
    Array<unsigned int> *indexArray =
      new Array<unsigned int>(1, cudaFlag, (unsigned int) nVertex);

    // Set equal to first dimension of vertexCoordinates
    vxTemp->SetEqualComb(connectivity->vertexCoordinates, 0, 0);
    indexArray->SetToSeries();
    vxTemp->SortByKey(indexArray);

    // Two leftmost and rightmost vertices
    unsigned int vLeft1, vLeft2, vRight1, vRight2;
    indexArray->GetSingleValue(&vLeft1, 0);
    indexArray->GetSingleValue(&vLeft2, 1);
    indexArray->GetSingleValue(&vRight1, nVertex - 1);
    indexArray->GetSingleValue(&vRight2, nVertex - 2);

    delete vxTemp;
    delete indexArray;

    // y-coordinates of leftmost vertices
    real vy1, vy2;
    real2 Vy;
    connectivity->vertexCoordinates->GetSingleValue(&Vy, vLeft1);
    vy1 = Vy.y;
    connectivity->vertexCoordinates->GetSingleValue(&Vy, vLeft2);
    vy2 = Vy.y;

    if (vy1 > vy2) {
      int temp = vLeft1;
      vLeft1 = vLeft2;
      vLeft2 = temp;
    }

    // y-coordinates of rightmost vertices
    connectivity->vertexCoordinates->GetSingleValue(&Vy, vRight1);
    vy1 = Vy.y;
    connectivity->vertexCoordinates->GetSingleValue(&Vy, vRight2);
    vy2 = Vy.y;

    if (vy1 > vy2) {
      int temp = vRight1;
      vRight1 = vRight2;
      vRight2 = temp;
    }

    // Find edge between two leftmost and to rightmost vertices
    Array<int> *edgeLeftRight = new Array<int>(1, cudaFlag, (unsigned int) 2);
    edgeLeftRight->SetToValue(-1);
    int *pEdgeLeftRight = edgeLeftRight->GetPointer();

    if (cudaFlag == 1) {
      int nBlocks = 128;
      int nThreads = 128;

      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                         devFindEdgeBetween,
                                         (size_t) 0, 0);

      devFindEdgeBetween<<<nBlocks, nThreads>>>
        (nTriangle, vLeft1, vLeft2, pTv, pTe, &(pEdgeLeftRight[0]));
      devFindEdgeBetween<<<nBlocks, nThreads>>>
        (nTriangle, vRight2, vRight1, pTv, pTe, &(pEdgeLeftRight[1]));

      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    } else {
      for (int i = 0; i < nTriangle; i++) {
        FindEdgeBetween(i, vLeft1, vLeft2, pTv, pTe, &(pEdgeLeftRight[0]));
        FindEdgeBetween(i, vRight2, vRight1, pTv, pTe, &(pEdgeLeftRight[1]));
      }
    }

    if (cudaFlag == 1)
      edgeLeftRight->TransformToHost();
    pEdgeLeftRight = edgeLeftRight->GetPointer();

    int eleft = pEdgeLeftRight[0];
    int eright = pEdgeLeftRight[1];

    delete edgeLeftRight;

    if (eleft == -1 || eright == -1) {
      std::cout << "Could not find edges to make domain periodic! "
                << eleft << " " << eright << std::endl;

      std::cout << vLeft1 << " " << vLeft2 << std::endl;
      std::cout << vRight1 << " " << vRight2 << std::endl;
      throw std::runtime_error("");
    }

    connectivity->triangleVertices->SetSize(nTriangle + 2);
    connectivity->triangleEdges->SetSize(nTriangle + 2);
    connectivity->edgeTriangles->SetSize(nEdge + 3);

    int3 V;
    V.x = (int) vLeft1;
    V.y = (int) vLeft2;
    V.z = (int) (vRight2 - nVertex);
    connectivity->triangleVertices->SetSingleValue(V, nTriangle);
    V.x = (int) vRight2;
    V.y = (int) vRight1;
    V.z = (int) (vLeft1 + nVertex);
    connectivity->triangleVertices->SetSingleValue(V, nTriangle + 1);

    V.x = (int) eleft;
    V.y = nEdge + 2;
    V.z = nEdge + 1;
    connectivity->triangleEdges->SetSingleValue(V, nTriangle);
    V.x = (int) eright;
    V.y = nEdge;
    V.z = nEdge + 1;
    connectivity->triangleEdges->SetSingleValue(V, nTriangle + 1);

    int et1Left, et2Left, et1Right, et2Right;
    int2 T;
    connectivity->edgeTriangles->GetSingleValue(&T, eleft);
    et1Left = T.x;
    et2Left = T.y;
    if (et1Left == -1) T.x = nTriangle;
    if (et2Left == -1) T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, eleft);

    connectivity->edgeTriangles->GetSingleValue(&T, eright);
    et1Right = T.x;
    et2Right = T.y;
    if (et1Right == -1) T.x = nTriangle + 1;
    if (et2Right == -1) T.y = nTriangle + 1;
    connectivity->edgeTriangles->SetSingleValue(T, eright);

    T.x = nTriangle + 1;
    T.y = -1;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge);
    T.x = nTriangle + 1;
    T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge + 1);
    T.x = -1;
    T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge + 2);

    nTriangle += 2;
    nEdge += 3;
  }

  // Make periodic in y
  if (meshParameter->periodicFlagY) {
    Array<real> *vyTemp =
      new Array<real>(1, cudaFlag, (unsigned int) nVertex);
    Array<unsigned int> *indexArray =
      new Array<unsigned int>(1, cudaFlag, (unsigned int) nVertex);

    // Set equal to second dimension of vertexCoordinates
    vyTemp->SetEqualComb(connectivity->vertexCoordinates, 0, 1);
    indexArray->SetToSeries();
    vyTemp->SortByKey(indexArray);

    // Two topmost and bottommost vertices
    unsigned int vBottom1, vBottom2, vTop1, vTop2;
    indexArray->GetSingleValue(&vBottom1, 0);
    indexArray->GetSingleValue(&vBottom2, 1);
    indexArray->GetSingleValue(&vTop1, nVertex - 1);
    indexArray->GetSingleValue(&vTop2, nVertex - 2);

    delete vyTemp;
    delete indexArray;

    // x-coordinates of bottommost vertices
    real vx1, vx2;
    real2 Vx;
    connectivity->vertexCoordinates->GetSingleValue(&Vx, vBottom1);
    vx1 = Vx.x;
    connectivity->vertexCoordinates->GetSingleValue(&Vx, vBottom2);
    vx2 = Vx.x;

    if (vx1 > vx2) {
      int temp = vBottom1;
      vBottom1 = vBottom2;
      vBottom2 = temp;
    }

    // x-coordinates of topmost vertices
    connectivity->vertexCoordinates->GetSingleValue(&Vx, vTop1);
    vx1 = Vx.x;
    connectivity->vertexCoordinates->GetSingleValue(&Vx, vTop2);
    vx2 = Vx.x;

    if (vx1 > vx2) {
      int temp = vTop1;
      vTop1 = vTop2;
      vTop2 = temp;
    }

    // Find edge between two bottommost and to topmost vertices
    Array<int> *edgeBottomTop = new Array<int>(1, cudaFlag, (unsigned int) 2);
    edgeBottomTop->SetToValue(-1);
    int *pEdgeBottomTop = edgeBottomTop->GetPointer();

    if (cudaFlag == 1) {
      int nBlocks = 128;
      int nThreads = 128;

      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                         devFindEdgeBetween,
                                         (size_t) 0, 0);

      devFindEdgeBetween<<<nBlocks, nThreads>>>
        (nTriangle, vBottom2, vBottom1, pTv, pTe, &(pEdgeBottomTop[0]));
      devFindEdgeBetween<<<nBlocks, nThreads>>>
        (nTriangle, vTop1, vTop2, pTv, pTe, &(pEdgeBottomTop[1]));

      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    } else {
      for (int i = 0; i < nTriangle; i++) {
        FindEdgeBetween(i, vBottom2, vBottom1, pTv, pTe, &(pEdgeBottomTop[0]));
        FindEdgeBetween(i, vTop1, vTop2, pTv, pTe, &(pEdgeBottomTop[1]));
      }
    }

    if (cudaFlag == 1)
      edgeBottomTop->TransformToHost();
    pEdgeBottomTop = edgeBottomTop->GetPointer();

    int eBottom = pEdgeBottomTop[0];
    int eTop = pEdgeBottomTop[1];

    delete edgeBottomTop;

    if (eBottom == -1 || eTop == -1) {
      std::cout << "Could not find y edges to make domain periodic! "
                << eBottom << " " << eTop << std::endl;

      std::cout << vBottom1 << " " << vBottom2 << std::endl;
      std::cout << vTop1 << " " << vTop2 << std::endl;
      throw std::runtime_error("");
    }

    connectivity->triangleVertices->SetSize(nTriangle + 2);
    connectivity->triangleEdges->SetSize(nTriangle + 2);
    connectivity->edgeTriangles->SetSize(nEdge + 3);

    int3 V;
    V.x = (int) vBottom2;
    V.y = (int) vBottom1;
    V.z = (int) (vTop2 - 3*nVertex);
    connectivity->triangleVertices->SetSingleValue(V, nTriangle);

    V.x = (int) vTop1;
    V.y = (int) vTop2;
    V.z = (int) (vBottom1 + 3*nVertex);
    connectivity->triangleVertices->SetSingleValue(V, nTriangle + 1);

    V.x = (int) eBottom;
    V.y = nEdge + 2;
    V.z = nEdge + 1;
    connectivity->triangleEdges->SetSingleValue(V, nTriangle);

    V.x = (int) eTop;
    V.y = nEdge + 2;
    V.z = nEdge;
    connectivity->triangleEdges->SetSingleValue(V, nTriangle + 1);

    int et1Bottom, et2Bottom, et1Top, et2Top;
    int2 T;
    connectivity->edgeTriangles->GetSingleValue(&T, eBottom);
    et1Bottom = T.x;
    et2Bottom = T.y;
    if (et1Bottom == -1) T.x = nTriangle;
    if (et2Bottom == -1) T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, eBottom);

    connectivity->edgeTriangles->GetSingleValue(&T, eTop);
    et1Top = T.x;
    et2Top = T.y;

    if (et1Top == -1) T.x = nTriangle + 1;
    if (et2Top == -1) T.y = nTriangle + 1;
    connectivity->edgeTriangles->SetSingleValue(T, eTop);

    T.x = nTriangle + 1;
    T.y = -1;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge);
    T.x = -1;
    T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge + 1);
    T.x = nTriangle + 1;
    T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge + 2);

    nTriangle += 2;
    nEdge += 3;
  }

  // Add two more triangles if periodic in both x and y
  if (meshParameter->periodicFlagX && meshParameter->periodicFlagY) {
    connectivity->triangleVertices->SetSize(nTriangle + 2);
    connectivity->triangleEdges->SetSize(nTriangle + 2);
    connectivity->edgeTriangles->SetSize(nEdge + 1);

    int e1, v1, v2;
    int3 V;
    int3 E;
    connectivity->triangleEdges->GetSingleValue(&E, nTriangle - 1);
    e1 = E.x;
    connectivity->triangleVertices->GetSingleValue(&V, nTriangle - 1);
    v1 = V.x;
    v2 = V.y;

    int2 T;
    connectivity->edgeTriangles->GetSingleValue(&T, e1);
    int et1e1 = T.x;
    int et2e1 = T.y;

    if (et1e1 != -1 && et2e1 != -1) {
      e1 = E.y;
      v1 = V.y;
      v2 = V.z;
    }
    connectivity->edgeTriangles->GetSingleValue(&T, e1);
    et1e1 = T.x;
    et2e1 = T.y;
    if (et1e1 != -1 && et2e1 != -1) {
      e1 = E.z;
      v1 = V.z;
      v2 = V.x;
    }

    int e2, v4;
    connectivity->triangleEdges->GetSingleValue(&E, nTriangle - 2);
    e2 = E.x;
    connectivity->triangleVertices->GetSingleValue(&V, nTriangle - 2);
    v4 = V.y;
    connectivity->edgeTriangles->GetSingleValue(&T, e2);
    int et1e2 = T.x;
    int et2e2 = T.y;
    if (et1e2 != -1 && et2e2 != -1) {
      e2 = E.y;
      v4 = V.z;
    }
    connectivity->edgeTriangles->GetSingleValue(&T, e2);
    et1e2 = T.x;
    et2e2 = T.y;
    if (et1e2 != -1 && et2e2 != -1) {
      e2 = E.z;
      v4 = V.x;
    }

    int e3, v5;
    connectivity->triangleEdges->GetSingleValue(&E, nTriangle - 4);
    e3 = E.x;
    connectivity->triangleVertices->GetSingleValue(&V, nTriangle - 4);
    v5 = V.y;
    connectivity->edgeTriangles->GetSingleValue(&T, e3);
    int et1e3 = T.x;
    int et2e3 = T.y;

    if (et1e3 != -1 && et2e3 != -1) {
      e3 = E.y;
      v5 = V.z;
    }
    connectivity->edgeTriangles->GetSingleValue(&T, e3);
    et1e3 = T.x;
    et2e3 = T.y;
    if (et1e3 != -1 && et2e3 != -1) {
      e3 = E.z;
      v5 = V.x;
    }

    int e4;
    connectivity->triangleEdges->GetSingleValue(&E, nTriangle - 3);
    e4 = E.x;
    connectivity->edgeTriangles->GetSingleValue(&T, e4);
    int et1e4 = T.x;
    int et2e4 = T.y;
    if (et1e4 != -1 && et2e4 != -1) e4 = E.y;
    connectivity->edgeTriangles->GetSingleValue(&T, e4);
    et1e4 = T.x;
    et2e4 = T.y;
    if (et1e4 != -1 && et2e4 != -1) e4 = E.z;

    V.x = v2;
    V.y = v1;
    V.z = v4 + 2*nVertex;
    connectivity->triangleVertices->SetSingleValue(V, nTriangle);

    E.x = e1;
    E.y = e4;
    E.z = nEdge;
    connectivity->triangleEdges->SetSingleValue(E, nTriangle);

    V.x = v5;
    V.y = v2;
    V.z = v4 + 2*nVertex;
    connectivity->triangleVertices->SetSingleValue(V, nTriangle + 1);

    E.x = e3;
    E.y = nEdge;
    E.z = e2;
    connectivity->triangleEdges->SetSingleValue(E, nTriangle + 1);

    T.x = nTriangle;  // 6
    T.y = nTriangle + 1;  // 7
    connectivity->edgeTriangles->SetSingleValue(T, nEdge);

    T.x = nTriangle - 2;  // 4
    T.y = nTriangle + 1;  // 7
    connectivity->edgeTriangles->SetSingleValue(T, e2);

    T.x = nTriangle - 1;  // 5
    T.y = nTriangle;  // 6
    connectivity->edgeTriangles->SetSingleValue(T, e1);

    T.x = nTriangle - 4;  // 2
    T.y = nTriangle + 1;  // 7
    connectivity->edgeTriangles->SetSingleValue(T, e3);

    T.x = nTriangle - 3;  // 3
    T.y = nTriangle;  // 6
    connectivity->edgeTriangles->SetSingleValue(T, e4);

    nTriangle += 2;
    nEdge += 1;
  }
}

}  // namespace astrix
