// -*-c++-*-
/*! \file addtoperiodic.cu
\brief Functions for adjusting indices of periodic vertices when adding new vertices*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//######################################################################
/*! \brief Correct indices of periodic vertices of triangle \a n \n

When adding \a nAdd vertices to the mesh, indices of periodic vertices will have to increase by \a nAdd (periodic in x) 2\a nAdd (periodic in y) or 3\a nAdd (periodic in x and y).

\param n Index of triangle to consider
\param *pTv Pointer to triangle vertices
\param nVertex Total number of vertices in Mesh (before adding \a nAdd)
\param nAdd Number of vertices to be added to Mesh*/
//######################################################################

__host__ __device__
void AddToPeriodicSingle(int n, int3 *pTv, int nVertex, int nAdd)
{
  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;

  if (a > 0) a += (a/nVertex)*nAdd;
  if (b > 0) b += (b/nVertex)*nAdd;
  if (c > 0) c += (c/nVertex)*nAdd;

  if (a < 0) a += ((a + 1)/nVertex - 1)*nAdd;
  if (b < 0) b += ((b + 1)/nVertex - 1)*nAdd;
  if (c < 0) c += ((c + 1)/nVertex - 1)*nAdd;

  pTv[n].x = a;
  pTv[n].y = b;
  pTv[n].z = c;
}

//######################################################################
/*! \brief Correct indices of periodic vertices of triangle \n

When adding \a nAdd vertices to the mesh, indices of periodic vertices will have to increase by \a nAdd (periodic in x) 2\a nAdd (periodic in y) or 3\a nAdd (periodic in x and y).

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param nVertex Total number of vertices in Mesh
\param nAdd Number of vertices to be added to Mesh*/
//######################################################################

__global__ void
devAddToPeriodic(int nTriangle, int3 *pTv, int nVertex, int nAdd)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    AddToPeriodicSingle(i, pTv, nVertex, nAdd);

    i += blockDim.x*gridDim.x;
  }
}

//##############################################################################
/*! When adding \a nAdd vertices to the mesh, indices of periodic vertices will have to increase by \a nAdd (periodic in x) 2\a nAdd (periodic in y) or 3\a nAdd (periodic in x and y).

\param *connectivity Pointer to basic Mesh data
\param nAdd Number of vertices to be added to Mesh*/
//##############################################################################
void Refine::AddToPeriodic(Connectivity * const connectivity,
                           const int nAdd)
{
  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  int3 *pTv = connectivity->triangleVertices->GetPointer();

  // Add nAdd to periodic vertices so that they remain >= nVertex
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAddToPeriodic,
                                       (size_t) 0, 0);

    devAddToPeriodic<<<nBlocks, nThreads>>>
      (nTriangle, pTv, nVertex, nAdd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      AddToPeriodicSingle(n, pTv, nVertex, nAdd);
  }
}

}
