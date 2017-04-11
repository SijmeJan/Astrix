// -*-c++-*-
/*! \file sedov.cu
\brief Diagnostics for Sedov blast wave test problem

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//######################################################################
//######################################################################

//__host__ __device__
void SetSedovSingle(int n, const real2 *pVc, realNeq *pState, real *eta, real *A,
                    int nSedov)
{
  real x = pVc[n].x;
  real y = pVc[n].y;

#if SEDOV_1D == 1
  real r = sqrt(x*x);
#else
  real r = sqrt(x*x + y*y);
#endif

  int i = 0;
  while (eta[i] < r) i++;

  int j = std::max(0, i - 1);
  real dens = A[i];
  if (i != j)
    dens =
      (eta[i] - r)*A[j]/(eta[i] - eta[j]) +
      (r - eta[j])*A[i]/(eta[i] - eta[j]);

  pState[n].x = dens;
}


//######################################################################
//######################################################################

void Simulation::SedovSetAnalytic(Array<realNeq> *state, real E)
{
  unsigned int nVertex = mesh->GetNVertex();

  realNeq *pState = state->GetPointer();

  const real2 *pVc = mesh->VertexCoordinatesData();
  real G = simulationParameter->specificHeatRatio;

  // Read in solution
#if SEDOV_CART == 1
  std::ifstream sedov("sedov1D.txt");
  if (!sedov.is_open()) {
    std::cout << "Error opening file sedov1D.txt" << std::endl;
    throw std::runtime_error("");
  }
#else
  std::ifstream sedov("sedov.txt");
  if (!sedov.is_open()) {
    std::cout << "Error opening file sedov.txt" << std::endl;
    throw std::runtime_error("");
  }
#endif

  int nSedov = 0;
  sedov >> nSedov;
  nSedov += 3;

  Array<real> *eta = new Array<real>(1, 0, nSedov);
  Array<real> *A = new Array<real>(1, 0, nSedov);

  real *pEta = eta->GetPointer();
  real *pA  = A->GetPointer();

  for (int j = 1; j < nSedov - 2; j++)
    sedov >> pEta[j] >> pA[j];

  pEta[0] = 0.0;
  pA[0] = pA[1];

  pEta[nSedov - 2] = pEta[nSedov - 3] + 1.0e-10;
  pA[nSedov - 2] = (G - 1.0)/(G + 1.0);
  pEta[nSedov - 1] = 1.0e10;
  pA[nSedov - 1] = (G - 1.0)/(G + 1.0);

#if SEDOV_CART == 1
  E = 0.25*E/mesh->GetPy();

  for (int j = 0; j < nSedov; j++) {
    pEta[j] = pEta[j]*std::pow(E*Sq(simulationTime + 1.0e-10), 1.0/3.0);
    pA[j] = pA[j]*(G + 1.0)/(G - 1.0);
  }
#else
  for (int j = 0; j < nSedov; j++) {
    pEta[j] = pEta[j]*std::pow(E*Sq(simulationTime + 1.0e-10), 0.25);
    pA[j] = pA[j]*(G + 1.0)/(G - 1.0);
  }
#endif

  sedov.close();

  /*
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSetSedov,
                                       (size_t) 0, 0);

    devSetSedov<<<nBlocks, nThreads>>>
      (nVertex, pVc, pState, eta, A, G);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
  */
    for (unsigned int n = 0; n < nVertex; n++)
      SetSedovSingle(n, pVc, pState, pEta, pA, nSedov);
    //}

    //int qq; std::cin >> qq;

  delete eta;
  delete A;
}


//######################################################################
//######################################################################

real Simulation::SedovShockPos()
{
  int nVertex = mesh->GetNVertex();
  int nTriangle = mesh->GetNTriangle();

  // Triangle vertex indices
  const int3 *pTv = mesh->TriangleVerticesData();
  const real2 *pVc = mesh->VertexCoordinatesData();

  // Inward pointing edge normals
  const real2 *pTn1 = mesh->TriangleEdgeNormalsData(0);
  const real2 *pTn2 = mesh->TriangleEdgeNormalsData(1);
  const real2 *pTn3 = mesh->TriangleEdgeNormalsData(2);

  // Edge lengths
  const real3 *pTl = mesh->TriangleEdgeLengthData();

  // State at vertices
  realNeq *pState = vertexState->GetPointer();

  real posTot = 0.0;
  int Ntot = 0;

  for (unsigned int n = 0; n < nTriangle; n++) {
    int a = pTv[n].x;
    int b = pTv[n].y;
    int c = pTv[n].z;

    int v = a;
    if (b < nVertex && b >= 0) v = b;
    if (c < nVertex && c >= 0) v = c;
    real x = pVc[v].x;
    real y = pVc[v].y;

    while (a >= nVertex) a -= nVertex;
    while (b >= nVertex) b -= nVertex;
    while (c >= nVertex) c -= nVertex;
    while (a < 0) a += nVertex;
    while (b < 0) b += nVertex;
    while (c < 0) c += nVertex;

    real d1 = pState[a].x;
    real d2 = pState[b].x;
    real d3 = pState[c].x;

    // Triangle edge lengths
    real tl1 = pTl[n].x;
    real tl2 = pTl[n].y;
    real tl3 = pTl[n].z;

    // Triangle inward pointing normals
    real nx1 = pTn1[n].x*tl1;
    real ny1 = pTn1[n].y*tl1;
    real nx2 = pTn2[n].x*tl2;
    real ny2 = pTn2[n].y*tl2;
    real nx3 = pTn3[n].x*tl3;
    real ny3 = pTn3[n].y*tl3;

    real drhodx = d1*nx1 + d2*nx2 + d3*nx3;
    real drhody = d1*ny1 + d2*ny2 + d3*ny3;

    real dmin = std::min(d1, std::min(d2, d3));
    real dmax = std::max(d1, std::max(d2, d3));

#if SEDOV_CART == 1
    if (dmin < 1.5 && dmax > 1.5 && drhodx < 0.0 && x > 0.0) {
      posTot = posTot + x;
      Ntot++;
    }
#else
    real r = sqrt(x*x + y*y);
    real drhodr = x*drhodx + y*drhody;
    if (dmin < 1.5 && dmax > 1.5 && drhodr < 0.0) {
      posTot = posTot + r;
      Ntot++;
    }
#endif
  }

  return posTot/((real) Ntot + 1.0e-30);
}

}  // namespace astrix
