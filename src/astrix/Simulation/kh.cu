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
/*! \brief Add KH eigenvector perturbation to single vertex

\param i Vertex to add perturbation to
\param *pVc Pointer to Array of vertex coordinates
\param *pState Pointer to Array of state vector
\param *dens Pointer to density perturbation Array (real and imaginary)
\param *velx Pointer to x velocity perturbation Array (real and imaginary)
\param *vely Pointer to y velocity perturbation Array (real and imaginary)
\param kxKH Number of wavelengths in x
\param *yKH Pointer to Array of y values of perturbation
\param miny Minimum y value in domain
\param maxy Maximum y value in domain
\param G Ratio of specific heats
\param G1 Ratio of specific heats - 1*/
//##############################################################################

__host__ __device__
void AddEigenVectorSingleKH(unsigned int i, const real2 *pVc, real4 *pState,
                            real2 *dens, real2 *velx, real2 *vely,
                            real kxKH, real *yKH,
                            real miny, real maxy, real G, real G1)
{
  real x = pVc[i].x;
  real y = pVc[i].y;

  if (y < miny) y += (maxy - miny);
  if (y > maxy) y -= (maxy - miny);

  real dyKH = yKH[1] - yKH[0];

  int jj = (int)((y - yKH[0])/dyKH);
#ifndef __CUDA_ARCH__
  if (jj < 0 || jj > 128) {
    std::cout << jj << " " << y << " " << yKH[0] << " " << yKH[129]
              << std::endl;
    int qq; std::cin >> qq;
  }
#endif

  real dRj = dens[jj].x + (y - yKH[jj])*(dens[jj + 1].x - dens[jj].x)/dyKH;
  real dIj = dens[jj].y + (y - yKH[jj])*(dens[jj + 1].y - dens[jj].y)/dyKH;
  real uRj = velx[jj].x + (y - yKH[jj])*(velx[jj + 1].x - velx[jj].x)/dyKH;
  real uIj = velx[jj].y + (y - yKH[jj])*(velx[jj + 1].y - velx[jj].y)/dyKH;
  real vRj = vely[jj].x + (y - yKH[jj])*(vely[jj + 1].x - vely[jj].x)/dyKH;
  real vIj = vely[jj].y + (y - yKH[jj])*(vely[jj + 1].y - vely[jj].y)/dyKH;

  real d0 = pState[i].x;
  real a0 = pState[i].y;
  real b0 = pState[i].z;
  real e0 = pState[i].w;
  real p0 = G1*(e0 - 0.5*(a0*a0 + b0*b0)/d0);
  /*
#ifndef __CUDA_ARCH__
  std::cout << "x = " << x
            << " y = " << y
            << " a0 = " << a0
            << " b0 = " << b0
            << " uRj = " << uRj
            << " uIj = " << uIj
            << " vRj = " << vRj
            << " vIj = " << vIj
            << std::endl;
  int qq; std::cin >> qq;
#endif
  */

  pState[i].x = d0 + dRj*cos(2.0*M_PI*kxKH*x) - dIj*sin(2.0*M_PI*kxKH*x);
  pState[i].y = a0 + d0*uRj*cos(2.0*M_PI*kxKH*x) - d0*uIj*sin(2.0*M_PI*kxKH*x);
  pState[i].z = b0 + d0*vRj*cos(2.0*M_PI*kxKH*x) - d0*vIj*sin(2.0*M_PI*kxKH*x);
  real pr = p0 + G*p0*(dRj*cos(2.0*M_PI*kxKH*x) - dIj*sin(2.0*M_PI*kxKH*x))/d0;
  pState[i].w = 0.5*(Sq(pState[i].y) + Sq(pState[i].z))/pState[i].x + pr/G1;
}

__host__ __device__
void AddEigenVectorSingleKH(unsigned int i, const real2 *pVc, real3 *pState,
                            real2 *dens, real2 *velx, real2 *vely,
                            real kxKH, real *yKH,
                            real miny, real maxy, real G, real G1)
{
  // Dummy function; no eigenvector to add if solving isothermal equation
}

__host__ __device__
void AddEigenVectorSingleKH(unsigned int i, const real2 *pVc, real *pState,
                            real2 *dens, real2 *velx, real2 *vely,
                            real kxKH, real *yKH,
                            real miny, real maxy, real G, real G1)
{
  // Dummy function; no eigenvector to add if solving scalar equation
}

//######################################################################
/*! \brief Kernel adding KH eigenvector perturbation

\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to Array of vertex coordinates
\param *pState Pointer to Array of state vector
\param *dens Pointer to density perturbation Array (real and imaginary)
\param *velx Pointer to x velocity perturbation Array (real and imaginary)
\param *vely Pointer to y velocity perturbation Array (real and imaginary)
\param kxKH Number of wavelengths in x
\param *yKH Pointer to Array of y values of perturbation
\param miny Minimum y value in domain
\param maxy Maximum y value in domain
\param G Ratio of specific heats
\param G1 Ratio of specific heats - 1*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devAddEigenVectorKH(unsigned int nVertex, const real2 *pVc, realNeq *pState,
                    real2 *dens, real2 *velx, real2 *vely,
                    real kxKH, real *yKH,
                    real miny, real maxy, real G, real G1)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    AddEigenVectorSingleKH(n, pVc, pState, dens, velx, vely,
                           kxKH, yKH, miny, maxy, G, G1);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Add eigenvector perturbation, specified in a file eigvec.txt. A runtime error is thrown if this file is not found.*/
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
  // Real and imaginary parts, so real2
  Array<real2> *dens = new Array<real2>(1, 0, nKH);
  Array<real2> *velx = new Array<real2>(1, 0, nKH);
  Array<real2> *vely = new Array<real2>(1, 0, nKH);

  real *pyKH = yKH->GetPointer();
  real2 *pDens  = dens->GetPointer();
  real2 *pVelx  = velx->GetPointer();
  real2 *pVely  = vely->GetPointer();

  for (int j = 1; j < nKH - 1; j++)
    KH >> pyKH[j]
       >> pDens[j].x >> pDens[j].y
       >> pVelx[j].x >> pVelx[j].y
       >> pVely[j].x >> pVely[j].y;

  KH.close();

  pyKH[0] = pyKH[1] - (pyKH[2] - pyKH[1]);
  pDens[0] = pDens[nKH - 2];
  pVelx[0] = pVelx[nKH - 2];
  pVely[0] = pVely[nKH - 2];

  pyKH[nKH - 1] = pyKH[nKH - 2] + (pyKH[2] - pyKH[1]);
  pDens[nKH - 1] = pDens[1];
  pVelx[nKH - 1] = pVelx[1];
  pVely[nKH - 1] = pVely[1];

  // Transform to device
  if (cudaFlag == 1) {
    yKH->TransformToDevice();
    dens->TransformToDevice();
    velx->TransformToDevice();
    vely->TransformToDevice();

    pyKH = yKH->GetPointer();
    pDens = dens->GetPointer();
    pVelx = velx->GetPointer();
    pVely = vely->GetPointer();
  }

  real miny = mesh->GetMinY();
  real maxy = mesh->GetMaxY();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAddEigenVectorKH<realNeq, CL>,
                                       (size_t) 0, 0);

    devAddEigenVectorKH<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, pVc, pState, pDens, pVelx, pVely, kxKH, pyKH,
       miny, maxy, G, G - 1.0);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      AddEigenVectorSingleKH(n, pVc, pState, pDens, pVelx, pVely,
                             kxKH, pyKH, miny, maxy, G, G - 1.0);
  }

  delete yKH;
  delete dens;
  delete velx;
  delete vely;
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::KHAddEigenVector();
template void Simulation<real, CL_BURGERS>::KHAddEigenVector();
template void Simulation<real3, CL_CART_ISO>::KHAddEigenVector();
template void Simulation<real4, CL_CART_EULER>::KHAddEigenVector();

}  // namespace astrix
