// -*-c++-*-
/*! \file save.cpp
\brief File containing functions to save / restore simulation.*/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cuda_runtime_api.h>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"

namespace astrix {
  
//#########################################################################
/*! Write the current state to disk, generating output files dens###.dat 
  (density), momx###.dat (x-momentum), momy###.dat (y-momentum) and 
  ener###.dat (total energy). Hashes indicate a 3-digit number constructed 
  from nSave. 

  \param nSave Output number, used to generate file names. */
//#########################################################################

void Simulation::Save(int nSave)
{
  std::cout << "Save #" << nSave << "...";

  std::ofstream outFile;
  int nVertex = mesh->GetNVertex();

  // Save mesh data
  mesh->Save(nSave);
  
  // Copy data to host
  if (cudaFlag == 1) vertexState->CopyToHost();
  
  Array<real> *dens = new Array<real>(1, 0, nVertex);
  Array<real> *momx = new Array<real>(1, 0, nVertex);
  Array<real> *momy = new Array<real>(1, 0, nVertex);
  Array<real> *ener = new Array<real>(1, 0, nVertex);
  real *pDens = dens->GetPointer();
  real *pMomx = momx->GetPointer();
  real *pMomy = momy->GetPointer();
  real *pEner = ener->GetPointer();

#if N_EQUATION == 1
  real *state = vertexState->GetHostPointer();
  const real2 *pVc = mesh->VertexCoordinatesData();

  if (problemDef == PROBLEM_VORTEX) {
    for (int n = 0; n < nVertex; n++) {
      pDens[n] = state[n];

#if BURGERS == 1
      real u0 = 1.0;
      real u1 = 0.0;
      real u2 = 0.0;
      real amp = 0.001;
      for (int i = 0; i < 2; i++) {
	for (int j = 0; j < 2; j++) {
	  // Start center location
	  real xStart = -1.0 + (real)i;
	  real yStart = -1.0 + (real)j;
	  
	  // Current centre location
	  real cx = xStart + simulationTime;
	  real cy = yStart + simulationTime;
	  
	  real x = pVc[n].x - cx;
	  real y = pVc[n].y - cy;
	  real r = sqrt(x*x + y*y);
	  
	  if (r < 0.25) {
	    u1 += amp*cos(2.0*M_PI*r)*cos(2.0*M_PI*r);
	    u2 += amp*amp*4.0*M_PI*simulationTime*(x + y)*cos(2.0*M_PI*r)*
	      cos(2.0*M_PI*r)*cos(2.0*M_PI*r)*sin(2.0*M_PI*r)/(r + 1.0e-30);
	  }
	}
      }

      pMomx[n] = fabs(state[n] - u0 - u1)*state[n];
      pMomy[n] = fabs(state[n] - u0 - u1 - u2)*state[n];
      pEner[n] = state[n];
#else
      real u0 = 1.0;
      real u1 = 0.0;
      
      for (int i = 0; i < 2; i++) {
	// Start center location
	real xStart = -0.5 + (real)i;
	real yStart = 0.5;
	
	// Current centre location
	real cx = xStart + simulationTime;
	real cy = yStart;
	  
	real x = pVc[n].x - cx;
	real y = pVc[n].y - cy;
	real r = sqrt(x*x + y*y);

	if (r <= (real) 0.25) {
	  u1 += cos(2.0*M_PI*r)*cos(2.0*M_PI*r);
	  /*
	  std::cout << pVc[n].x << " "
		    << pVc[n].y << " "
		    << u0 + u1 << " "
		    << state[n] << std::endl;
	  int qq; std::cin >> qq;
	  */
	}

      }

      pMomx[n] = fabs(state[n] - u0 - u1)*state[n];
      pMomy[n] = state[n];
      pEner[n] = state[n];
#endif
    }
  }
#endif
#if N_EQUATION == 4
  real4 *state = vertexState->GetHostPointer();
  for (int n = 0; n < nVertex; n++) {
    pDens[n] = state[n].x;
    pMomx[n] = state[n].y;
    pMomy[n] = state[n].z;
    pEner[n] = state[n].w;
  }
#endif
  
  char fname[13];
  int sizeOfData = sizeof(real);
    
  // Write density binary
  snprintf(fname, sizeof(fname), "dens%4.4d.dat", nSave);
  outFile.open(fname, std::ios::binary);
  outFile.write(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  outFile.write(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  outFile.write(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  outFile.write(reinterpret_cast<char*>(pDens), nVertex*sizeof(real));
  outFile.close();

  // Write x-momentum binary
  snprintf(fname, sizeof(fname), "momx%4.4d.dat", nSave);
  outFile.open(fname, std::ios::binary);
  outFile.write(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  outFile.write(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  outFile.write(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  outFile.write(reinterpret_cast<char*>(pMomx), nVertex*sizeof(real));
  outFile.close();
  
  // Write y-momentum binary
  snprintf(fname, sizeof(fname), "momy%4.4d.dat", nSave);
  outFile.open(fname, std::ios::binary);
  outFile.write(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  outFile.write(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  outFile.write(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  outFile.write(reinterpret_cast<char*>(pMomy), nVertex*sizeof(real));
  outFile.close();
  
  // Write energy binary
  snprintf(fname, sizeof(fname), "ener%4.4d.dat", nSave);
  outFile.open(fname, std::ios::binary);
  outFile.write(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  outFile.write(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  outFile.write(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  outFile.write(reinterpret_cast<char*>(pEner), nVertex*sizeof(real));
  outFile.close();

  // Write blend factor
  int nTriangle = mesh->GetNTriangle();
  CalcShockSensor();
  if (cudaFlag == 1)
    triangleShockSensor->CopyToHost();
  real *pBlnd = triangleShockSensor->GetHostPointer();
  snprintf(fname, sizeof(fname), "blnd%4.4d.dat", nSave);
  outFile.open(fname, std::ios::binary);
  outFile.write(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  outFile.write(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  outFile.write(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  outFile.write(reinterpret_cast<char*>(pBlnd), nTriangle*sizeof(real));
  outFile.close();

  delete dens;
  delete momx;
  delete momy;
  delete ener;
    
  // Output save number so that we can restore latest save if wanted
  outFile.open("lastsave.txt");
  outFile << nSave << std::endl;
  outFile.close();
  
  std::cout << " Done" << std::endl;
}

//#########################################################################
/*! Restore state from previous save.
  \param nSave Save number to restore.*/
//#########################################################################

int Simulation::Restore(int nSave)
{
  std::ifstream inFile;
  char fname[13];
  int sizeOfData = sizeof(real);

  if (nSave == -1) {
    inFile.open("lastsave.txt");
    if (!inFile) {
      std::cout << "Could not open lastsave.txt, starting new simulation"
		<< std::endl;
      return 0;
    } 
    inFile >> nSave;
    inFile.close();
  }

  std::cout << "Restoring save #" << nSave << std::endl;

  mesh->ReadFromDisk(nSave);

  int nVertex = mesh->GetNVertex();
  int nTriangle = mesh->GetNTriangle();

  vertexState->SetSize(nVertex);
  vertexStateOld->SetSize(nVertex);
  vertexPotential->SetSize(nVertex);
  vertexParameterVector->SetSize(nVertex);
  vertexStateDiff->SetSize(nVertex);
  
  triangleResidueN->SetSize(nTriangle);
  triangleResidueLDA->SetSize(nTriangle);
  triangleResidueTotal->SetSize(nTriangle);
  if (intScheme == SCHEME_B || intScheme == SCHEME_BX)
    triangleBlendFactor->SetSize(nTriangle);

  CalcPotential();
  
  // Copy data to host
  if (cudaFlag == 1) vertexState->CopyToHost();
  
  Array<real> *dens = new Array<real>(1, 0, nVertex);
  Array<real> *momx = new Array<real>(1, 0, nVertex);
  Array<real> *momy = new Array<real>(1, 0, nVertex);
  Array<real> *ener = new Array<real>(1, 0, nVertex);
  real *pDens = dens->GetPointer();
  real *pMomx = momx->GetPointer();
  real *pMomy = momy->GetPointer();
  real *pEner = ener->GetPointer();
  
  // Read density binary
  snprintf(fname, sizeof(fname), "dens%4.4d.dat", nSave);
  inFile.open(fname, std::ios::binary);
  if (!inFile) {
    std::cout << "Could not open " << fname << ", aborting restart"
	      << std::endl;
    //return -1;
    throw std::runtime_error("");
  } 
  inFile.read(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  inFile.read(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  inFile.read(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  inFile.read(reinterpret_cast<char*>(pDens), nVertex*sizeof(real));
  inFile.close();

  // Read x momentum binary
  snprintf(fname, sizeof(fname), "momx%4.4d.dat", nSave);
  inFile.open(fname, std::ios::binary);
  if (!inFile) {
    std::cout << "Could not open " << fname  << ", aborting restart"
	      << std::endl;
    //return -1;
    throw std::runtime_error("");
  } 
  inFile.read(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  inFile.read(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  inFile.read(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  inFile.read(reinterpret_cast<char*>(pMomx), nVertex*sizeof(real));
  inFile.close();

  // Read y momentum binary
  snprintf(fname, sizeof(fname), "momy%4.4d.dat", nSave);
  inFile.open(fname, std::ios::binary);
  if (!inFile) {
    std::cout << "Could not open " << fname  << ", aborting restart"
	      << std::endl;
    //return -1;
    throw std::runtime_error("");
  } 
  inFile.read(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  inFile.read(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  inFile.read(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  inFile.read(reinterpret_cast<char*>(pMomy), nVertex*sizeof(real));
  inFile.close();
  
  // Read energy binary
  snprintf(fname, sizeof(fname), "ener%4.4d.dat", nSave);
  inFile.open(fname, std::ios::binary);
  if (!inFile) {
    std::cout << "Could not open " << fname  << ", aborting restart"
	      << std::endl;
    //return -1;
    throw std::runtime_error("");
  } 
  inFile.read(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  inFile.read(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  inFile.read(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  inFile.read(reinterpret_cast<char*>(pEner), nVertex*sizeof(real));
  inFile.close();

#if N_EQUATION == 1
  real *state = vertexState->GetHostPointer();
  for (int n = 0; n < nVertex; n++) {
    state[n] = pDens[n];
  }
#endif
#if N_EQUATION == 4
  real4 *state = vertexState->GetHostPointer();
  for (int n = 0; n < nVertex; n++) {
    state[n].x = pDens[n];
    state[n].y = pMomx[n];
    state[n].z = pMomy[n];
    state[n].w = pEner[n];
  }
#endif
  // Copy data to device
  if (cudaFlag == 1) vertexState->CopyToDevice();

  std::cout << "Done restoring" << std::endl;
  
  return nSave + 1;
}

//#########################################################################
/*! Do a fine grain save, i.e. write output files for certain global quantities but do not do a full data dump.  

  \param nSave Output number, used to determine whether to create new file. */
//#########################################################################

void Simulation::FineGrainSave(int nSave)
{
  std::ofstream outFile;
  int nVertex = mesh->GetNVertex();

  if (nSave == 0) outFile.open("simulation.dat");
  else outFile.open("simulation.dat", std::ios::app);
  outFile << std::setprecision(10)
	  << simulationTime << " "
	  << nVertex << " "
	  << TotalMass() << std::endl;
  outFile.close();

  if (problemDef == PROBLEM_KH) {
    real M = 0.0, E = 0.0;
    KHDiagnostics(M, E);

    if (nSave == 0) outFile.open("kh.dat");
    else outFile.open("kh.dat", std::ios::app);
    outFile << std::setprecision(10)
	    << simulationTime << " "
	    << M << " "
	    << E << std::endl;
    outFile.close();
  }
}

}
