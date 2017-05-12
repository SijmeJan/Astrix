// -*-c++-*-
/*! \file rt.cu
\brief Diagnostics for Rayleigh-Taylor test problem

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
/*! \brief Add RT eigenvector perturbation to single vertex

\param i Vertex to add perturbation to
\param *pVc Pointer to Array of vertex coordinates
\param *pState Pointer to Array of state vector
\param *pVp Pointer to external potential at vertices
\param *dens Pointer to density perturbation Array (real and imaginary)
\param *velx Pointer to x velocity perturbation Array (real and imaginary)
\param *vely Pointer to y velocity perturbation Array (real and imaginary)
\param *pres Pointer to pressure perturbation Array (real and imaginary)
\param kxRT Number of wavelengths in x
\param *yRT Pointer to Array of y values of perturbation
\param miny Minimum y value in domain
\param maxy Maximum y value in domain
\param G Ratio of specific heats
\param G1 Ratio of specific heats - 1
\param time Current time
\param omega2 Frequency squared of eigenmode*/
//##############################################################################

__host__ __device__
void AddEigenVectorSingleRT(unsigned int i, const real2 *pVc, real4 *pState,
                            real *pVp,
                            real2 *dens, real2 *velx, real2 *vely, real2 *pres,
                            real kxRT, real *yRT,
                            real miny, real maxy, real G, real G1,
                            real time, real omega2)
{
  real x = pVc[i].x;
  real y = pVc[i].y;

  // Find index to interpolate from
  int jj = 0;
  while (y > yRT[jj]) jj++;

  // Interpolate density, velocity and pressure
  real dyRT = yRT[jj + 1] - yRT[jj];
  real dRj = dens[jj].x + (y - yRT[jj])*(dens[jj + 1].x - dens[jj].x)/dyRT;
  real dIj = dens[jj].y + (y - yRT[jj])*(dens[jj + 1].y - dens[jj].y)/dyRT;
  real uRj = velx[jj].x + (y - yRT[jj])*(velx[jj + 1].x - velx[jj].x)/dyRT;
  real uIj = velx[jj].y + (y - yRT[jj])*(velx[jj + 1].y - velx[jj].y)/dyRT;
  real vRj = vely[jj].x + (y - yRT[jj])*(vely[jj + 1].x - vely[jj].x)/dyRT;
  real vIj = vely[jj].y + (y - yRT[jj])*(vely[jj + 1].y - vely[jj].y)/dyRT;
  real pRj = pres[jj].x + (y - yRT[jj])*(pres[jj + 1].x - pres[jj].x)/dyRT;
  real pIj = pres[jj].y + (y - yRT[jj])*(pres[jj + 1].y - pres[jj].y)/dyRT;

  real amp = 0.0;//1.0e-4;

  // Background state
  real d0 = pState[i].x;
  real a0 = pState[i].y;
  real b0 = pState[i].z;
  real e0 = pState[i].w;
  real p0 = G1*(e0 - 0.5*(a0*a0 + b0*b0)/d0 - d0*pVp[i]);

  real f = M_PI*kxRT*x;
  if (omega2 > 0.0)
    f -= sqrt(omega2)*time;

  // Add perturbation
  pState[i].x = d0 + amp*(dRj*cos(f) - dIj*sin(f));
  pState[i].y = a0 + d0*amp*(uRj*cos(f) - uIj*sin(f));
  pState[i].z = b0 + d0*amp*(vRj*cos(f) - vIj*sin(f));
  real pr = p0 + amp*(pRj*cos(f) - pIj*sin(f));
  pState[i].w = 0.5*(Sq(pState[i].y) + Sq(pState[i].z))/pState[i].x +
    pState[i].x*pVp[i] + pr/G1;
}

__host__ __device__
void AddEigenVectorSingleRT(unsigned int i, const real2 *pVc, real3 *pState,
                            real *pVp,
                            real2 *dens, real2 *velx, real2 *vely, real2 *pres,
                            real kxRT, real *yRT,
                            real miny, real maxy, real G, real G1,
                            real time, real omega2)
{
  // Dummy function; no eigenvector to add if solving isothermal equation
}

__host__ __device__
void AddEigenVectorSingleRT(unsigned int i, const real2 *pVc, real *pState,
                            real *pVp,
                            real2 *dens, real2 *velx, real2 *vely, real2 *pres,
                            real kxRT, real *yRT,
                            real miny, real maxy, real G, real G1,
                            real time, real omega2)
{
  // Dummy function; no eigenvector to add if solving scalar equation
}

//######################################################################
/*! \brief Kernel adding RT eigenvector perturbation

\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to Array of vertex coordinates
\param *pState Pointer to Array of state vector
\param *pVp Pointer to external potential at vertices
\param *dens Pointer to density perturbation Array (real and imaginary)
\param *velx Pointer to x velocity perturbation Array (real and imaginary)
\param *vely Pointer to y velocity perturbation Array (real and imaginary)
\param *pres Pointer to pressure perturbation Array (real and imaginary)
\param kxRT Number of wavelengths in x
\param *yRT Pointer to Array of y values of perturbation
\param miny Minimum y value in domain
\param maxy Maximum y value in domain
\param G Ratio of specific heats
\param G1 Ratio of specific heats - 1*/
//######################################################################

__global__ void
devAddEigenVectorRT(unsigned int nVertex, const real2 *pVc, realNeq *pState,
                    real *pVp,
                    real2 *dens, real2 *velx, real2 *vely, real2 *pres,
                    real kxRT, real *yRT,
                    real miny, real maxy, real G, real G1,
                    real time, real omega2)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    AddEigenVectorSingleRT(n, pVc, pState, pVp, dens, velx, vely, pres,
                           kxRT, yRT, miny, maxy, G, G1, time, omega2);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Add eigenvector perturbation, specified in a file eigvec.txt. A runtime error is thrown if this file is not found.*/
//######################################################################

void Simulation::RTAddEigenVector()
{
  unsigned int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();
  real *pVp = vertexPotential->GetPointer();
  real G = simulationParameter->specificHeatRatio;

  const real2 *pVc = mesh->VertexCoordinatesData();

  // Read in RT eigenvector
  std::ifstream RT("eigvec.txt");
  if (!RT.is_open()) {
    std::cout << "Error opening file " << "eigvec.txt" << std::endl;
    throw std::runtime_error("");
  }

  // Get number of points in y and number of wavelengths in x
  int nRT = 128 + 3;
  RT >> nRT;
  // Need to make array bigger to ease interpolation
  nRT += 3;
  real kxRT = 1.0;
  RT >> kxRT;
  real omega2 = 0.0;
  RT >> omega2;

  //std::cout << omega2 << " " << 2.0*M_PI/sqrt(omega2) << std::endl;
  //int qq; std::cin >> qq;

  // Create arrays of correct size on host
  Array<real> *yRT = new Array<real>(1, 0, nRT);
  // Real and imaginary parts, so real2
  Array<real2> *dens = new Array<real2>(1, 0, nRT);
  Array<real2> *velx = new Array<real2>(1, 0, nRT);
  Array<real2> *vely = new Array<real2>(1, 0, nRT);
  Array<real2> *pres = new Array<real2>(1, 0, nRT);

  real *pyRT = yRT->GetPointer();
  real2 *pDens  = dens->GetPointer();
  real2 *pVelx  = velx->GetPointer();
  real2 *pVely  = vely->GetPointer();
  real2 *pPres  = pres->GetPointer();

  // Read in data from file
  for (int j = 1; j < nRT - 1; j++)
    RT >> pyRT[j]
       >> pDens[j].x >> pDens[j].y
       >> pVelx[j].x >> pVelx[j].y
       >> pVely[j].x >> pVely[j].y
       >> pPres[j].x >> pPres[j].y;

  // Fill boundary values to ease interpolation
  pyRT[0] = pyRT[1] - (pyRT[2] - pyRT[1]);
  pDens[0] = pDens[1];
  pVelx[0] = pVelx[1];
  pVely[0] = pVely[1];
  pPres[0] = pPres[1];

  pyRT[nRT - 1] = pyRT[nRT - 2] + (pyRT[nRT - 2] - pyRT[nRT - 3]);
  pDens[nRT - 1] = pDens[nRT - 2];
  pVelx[nRT - 1] = pVelx[nRT - 2];
  pVely[nRT - 1] = pVely[nRT - 2];
  pPres[nRT - 1] = pPres[nRT - 2];

  RT.close();

  // Transform to device
  if (cudaFlag == 1) {
    yRT->TransformToDevice();
    dens->TransformToDevice();
    velx->TransformToDevice();
    vely->TransformToDevice();
    pres->TransformToDevice();

    pyRT = yRT->GetPointer();
    pDens = dens->GetPointer();
    pVelx = velx->GetPointer();
    pVely = vely->GetPointer();
    pPres = pres->GetPointer();
  }

  real miny = mesh->GetMinY();
  real maxy = mesh->GetMaxY();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAddEigenVectorRT,
                                       (size_t) 0, 0);

    devAddEigenVectorRT<<<nBlocks, nThreads>>>
      (nVertex, pVc, pState, pVp, pDens, pVelx, pVely, pPres,
       kxRT, pyRT, miny, maxy, G, G - 1.0, simulationTime, omega2);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      AddEigenVectorSingleRT(n, pVc, pState, pVp, pDens, pVelx, pVely, pPres,
                             kxRT, pyRT, miny, maxy,
                             G, G - 1.0, simulationTime, omega2);
  }

  delete yRT;
  delete dens;
  delete velx;
  delete vely;
  delete pres;
}

}  // namespace astrix
