// -*-c++-*-
/*! \file values.cu
\brief Functions for calculating Morton values*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./morton.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {
  
//#########################################################################
/*! \brief Bit interleave operation needed for Morton-ordering

\param x X-coordinate
\param y Y-coordinate*/
//#########################################################################

__host__ __device__
unsigned int BitInterleave(unsigned short x, unsigned short y)
{
  unsigned int z = 0; // z gets the resulting 32-bit Morton Number.
  
  for (unsigned int i = 0; i < sizeof(x) * 8; i++)
    z |= (x & 1 << i) << i | (y & 1 << i) << (i + 1);

  return z;
}
  
//######################################################################
/*! \brief Kernel calculating Morton values based on vertex coordinates

\param nVertex Total number of vertices in Mesh
\param *mortValues Pointer to output array, will contain Morton values
\param *pVc Pointer to vertex coordinates
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary*/
//######################################################################

__global__ void 
devCalcMortonValuesVertex(int nVertex, unsigned int *mortValues, real2 *pVc,
			  real minx, real maxx, real miny, real maxy)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nVertex) {
    unsigned short x = 65533*(pVc[i].x - minx)/(maxx - minx) + 1;
    unsigned short y = 65533*(pVc[i].y - miny)/(maxy - miny) + 1;
    mortValues[i] = BitInterleave(x, y);
    
    i += gridDim.x*blockDim.x;
  }
}
  
//######################################################################
/*! Calculate Morton values for every vertex, based on their coordinates.

\param *connectivity Pointer to basic Mesh data*/
//######################################################################

void Morton::CalcValues(Connectivity * const connectivity)
{
  int nVertex = connectivity->vertexCoordinates->GetSize();
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  unsigned int *pVmort = vertexMorton->GetPointer();

  // Calculate Morton values
  if (cudaFlag == 1) {    
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devCalcMortonValuesVertex,
				       (size_t) 0, 0);

    devCalcMortonValuesVertex<<<nBlocks, nThreads>>>
      (nVertex, pVmort, pVc, minx, maxx, miny, maxy);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++) {
      unsigned short x = 65533*(pVc[i].x - minx)/(maxx - minx) + 1;
      unsigned short y = 65533*(pVc[i].y - miny)/(maxy - miny) + 1;
      pVmort[i] = BitInterleave(x, y);
    }
  }
}

}
