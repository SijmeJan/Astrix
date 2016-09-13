// -*-c++-*-
/*! \file flagsegment.cu
\brief Functions for determining how many and which points to be added will be placed on segments*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//######################################################################
/*! \brief Flag whether vertex \i will be placed on segment

\param i Index of point to be added
\param *pEdgeAdd Pointer to array containing edges onto which to place points
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pOnSegmentFlag Pointer to output array, will be 1 if \i will be placed on segment, 0 otherwise*/
//######################################################################

__host__ __device__
void FlagOnSegmentSingle(int i, int *pElementAdd, int2 *pEt,
			 unsigned int *pOnSegmentFlag, int nTriangle)
{
  int e = pElementAdd[i] - nTriangle;
  unsigned int ret = 0;
  if (e >= 0) 
    ret = pEt[e].x == -1 || pEt[e].y == -1;
  pOnSegmentFlag[i] = ret;
}
    
//######################################################################
/*! \brief Kernel flagging whether vertices will be placed on segment

\param nRefine Number of points to be added
\param *pEdgeAdd Pointer to array containing edges onto which to place points
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pOnSegmentFlag Pointer to output array, will be 1 if \i will be placed on segment, 0 otherwise*/
//######################################################################

__global__ void
devFlagOnSegment(int nRefine, int *pElementAdd, int2 *pEt,
		 unsigned int *pOnSegmentFlag, int nTriangle)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    FlagOnSegmentSingle(i, pElementAdd, pEt, pOnSegmentFlag, nTriangle);
  
    i += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! Find which of the points to be added will be placed on segments. This is not difficult using the \a edgeAdd Array, which contains the indices of edges to place points on. All we have to do is find out which of these edges are in fact segments. Returns total number of points to be placed on segments.

\param *onSegmentFlagScan Pointer to output array: a scanned version of the Array containing flags*/
//######################################################################

int Refine::FlagSegment(Connectivity * const connectivity,
			Array<unsigned int> * const onSegmentFlagScan)
{
  int nRefine = elementAdd->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();
  int *pElementAdd = elementAdd->GetPointer();
  
  Array<unsigned int> *onSegmentFlag = 
    new Array<unsigned int>(1, cudaFlag, (unsigned int) nRefine);

  // Flag points to be inserted on segment
  unsigned int *pOnSegmentFlag = onSegmentFlag->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFlagOnSegment, 
				       (size_t) 0, 0);

    devFlagOnSegment<<<nBlocks, nThreads>>>
      (nRefine, pElementAdd, pEt, pOnSegmentFlag, nTriangle);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (int i = 0; i < nRefine; i++) 
      FlagOnSegmentSingle(i, pElementAdd, pEt, pOnSegmentFlag, nTriangle);
  }
  
  onSegmentFlagScan->SetSize(nRefine);

  int nOnSegment = onSegmentFlag->ExclusiveScan(onSegmentFlagScan);

  delete onSegmentFlag;

  return nOnSegment;
}

}
