// -*-c++-*-
/*! \file kh.cu
\brief Diagnostics for Kelvin-Helmholtz test problem

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>
#include <fstream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//##############################################################################
//##############################################################################

__host__ __device__
void AddEigenVectorSingle(unsigned int i, const real2 *pVc, real4 *pState,
                          real *dR, real*dI,
                          real *uR, real *uI,
                          real *vR, real *vI,
                          real dyKH, real kxKH, real *yKH,
                          real miny, real maxy, real G, real G1)
{
  real x = pVc[i].x;
  real y = pVc[i].y;

  if (y < miny) y += (maxy - miny);
  if (y > maxy) y -= (maxy - miny);

  int jj = (int)((y - yKH[0])/dyKH);
#ifndef __CUDA_ARCH__
  if (jj < 0 || jj > 128) {
    std::cout << jj << " " << y << " " << yKH[0] << " " << yKH[129]
              << std::endl;
    int qq; std::cin >> qq;
  }
#endif

  real dRj = dR[jj] + (y - yKH[jj])*(dR[jj + 1] - dR[jj])/dyKH;
  real dIj = dI[jj] + (y - yKH[jj])*(dI[jj + 1] - dI[jj])/dyKH;
  real uRj = uR[jj] + (y - yKH[jj])*(uR[jj + 1] - uR[jj])/dyKH;
  real uIj = uI[jj] + (y - yKH[jj])*(uI[jj + 1] - uI[jj])/dyKH;
  real vRj = vR[jj] + (y - yKH[jj])*(vR[jj + 1] - vR[jj])/dyKH;
  real vIj = vI[jj] + (y - yKH[jj])*(vI[jj + 1] - vI[jj])/dyKH;

  real d0 = pState[i].x;
  real a0 = pState[i].y;
  real b0 = pState[i].z;
  real e0 = pState[i].w;
  real p0 = G1*(e0 - 0.5*(a0*a0 + b0*b0)/d0);

  pState[i].x = d0 + dRj*cos(2.0*M_PI*kxKH*x) - dIj*sin(2.0*M_PI*kxKH*x);
  pState[i].y = a0 + d0*uRj*cos(2.0*M_PI*kxKH*x) - d0*uIj*sin(2.0*M_PI*kxKH*x);
  pState[i].z = b0 + d0*vRj*cos(2.0*M_PI*kxKH*x) - d0*vIj*sin(2.0*M_PI*kxKH*x);
  real pr = p0 + G*p0*(dRj*cos(2.0*M_PI*kxKH*x) - dIj*sin(2.0*M_PI*kxKH*x))/d0;
  pState[i].w = 0.5*(Sq(pState[i].y) + Sq(pState[i].z))/pState[i].x + pr/G1;
}

__host__ __device__
void AddEigenVectorSingle(unsigned int i, const real2 *pVc, real3 *pState,
                          real *dR, real*dI,
                          real *uR, real *uI,
                          real *vR, real *vI,
                          real dyKH, real kxKH, real *yKH,
                          real miny, real maxy, real G, real G1)
{
  // Dummy function; no eigenvector to add if solving isothermal equation
}

__host__ __device__
void AddEigenVectorSingle(unsigned int i, const real2 *pVc, real *pState,
                          real *dR, real*dI,
                          real *uR, real *uI,
                          real *vR, real *vI,
                          real dyKH, real kxKH, real *yKH,
                          real miny, real maxy, real G, real G1)
{
  // Dummy function; no eigenvector to add if solving scalar equation
}

//######################################################################
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devAddEigenVector(unsigned int nVertex, const real2 *pVc, realNeq *pState,
                  real *dR, real*dI, real *uR, real *uI, real *vR, real *vI,
                  real dyKH, real kxKH, real *yKH,
                  real miny, real maxy, real G, real G1)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    AddEigenVectorSingle(n, pVc, pState, dR, dI, uR, uI, vR, vI,
                         dyKH, kxKH, yKH, miny, maxy, G, G1);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::KHAddEigenVector()
{
  unsigned int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();
  real G = simulationParameter->specificHeatRatio;

  const real2 *pVc = mesh->VertexCoordinatesData();

  // Read in KH eigenvector
  std::ifstream KH("eigvec.txt");
  if (!KH.is_open()) {
    std::cout << "Error opening file " << "eigvec.txt" << std::endl;
    throw std::runtime_error("");
  }

  int nKH = 128 + 2;
  KH >> nKH;
  nKH += 2;
  real kxKH = 1.0;
  KH >> kxKH;

  Array<real> *yKH = new Array<real>(1, 0, nKH);
  Array<real> *densReal = new Array<real>(1, 0, nKH);
  Array<real> *densImag = new Array<real>(1, 0, nKH);
  Array<real> *velxReal = new Array<real>(1, 0, nKH);
  Array<real> *velxImag = new Array<real>(1, 0, nKH);
  Array<real> *velyReal = new Array<real>(1, 0, nKH);
  Array<real> *velyImag = new Array<real>(1, 0, nKH);

  real *pyKH = yKH->GetPointer();
  real *pdR  = densReal->GetPointer();
  real *pdI  = densImag->GetPointer();
  real *puR  = velxReal->GetPointer();
  real *puI  = velxImag->GetPointer();
  real *pvR  = velyReal->GetPointer();
  real *pvI  = velyImag->GetPointer();

  for (int j = 1; j < nKH - 1; j++)
    KH >> pyKH[j] >> pdR[j] >> pdI[j] >> puR[j] >> puI[j] >> pvR[j] >> pvI[j];

  KH.close();

  pyKH[0] = pyKH[1] - (pyKH[2] - pyKH[1]);
  pdR[0] = pdR[nKH - 2];
  pdI[0] = pdI[nKH - 2];
  puR[0] = puR[nKH - 2];
  puI[0] = puI[nKH - 2];
  pvR[0] = pvR[nKH - 2];
  pvI[0] = pvI[nKH - 2];

  pyKH[nKH - 1] = pyKH[nKH - 2] + (pyKH[2] - pyKH[1]);
  pdR[nKH - 1] = pdR[1];
  pdI[nKH - 1] = pdI[1];
  puR[nKH - 1] = puR[1];
  puI[nKH - 1] = puI[1];
  pvR[nKH - 1] = pvR[1];
  pvI[nKH - 1] = pvI[1];

  real miny = mesh->GetMinY();
  real maxy = mesh->GetMaxY();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAddEigenVector<realNeq, CL>,
                                       (size_t) 0, 0);

    devAddEigenVector<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, pVc, pState, pdR, pdI, puR, puI, pvR, pvI,
       pyKH[1] - pyKH[0], kxKH, pyKH, miny, maxy, G, G - 1.0);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      AddEigenVectorSingle(n, pVc, pState, pdR, pdI, puR, puI, pvR, pvI,
                           pyKH[1] - pyKH[0], kxKH, pyKH, miny, maxy,
                           G, G - 1.0);
  }

  delete yKH;
  delete densReal;
  delete densImag;
  delete velxReal;
  delete velxImag;
  delete velyReal;
  delete velyImag;
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::KHAddEigenVector();
template void Simulation<real, CL_BURGERS>::KHAddEigenVector();
template void Simulation<real3, CL_CART_ISO>::KHAddEigenVector();
template void Simulation<real4, CL_CART_EULER>::KHAddEigenVector();

}  // namespace astrix
