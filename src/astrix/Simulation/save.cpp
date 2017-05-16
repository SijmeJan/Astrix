// -*-c++-*-
/*! \file save.cpp
\brief File containing functions to save / restore simulation.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/nvtxEvent.h"
#include "./VTK/vtk.h"
#include "./Param/simulationparameter.h"
#include "./Diagnostics/diagnostics.h"
#include "../Common/state.h"

namespace astrix {

//#########################################################################
/*! Write the current state to disk, generating output files dens###.dat
  (density), momx###.dat (x-momentum), momy###.dat (y-momentum) and
  ener###.dat (total energy). Hashes indicate a 3-digit number constructed
  from nSave.*/
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::Save()
{
  nvtxEvent *nvtxSave = new nvtxEvent("Save", 3);

  // Write VTK output
  if (simulationParameter->writeVTK == 1) {
    ReplaceEnergyWithPressure();
    char VTKname[15];
    snprintf(VTKname, sizeof(VTKname), "astrix%4.4d.vtk", nSave);
    VTK *vtk = new VTK();
    vtk->Write<realNeq, CL>(VTKname, mesh, vertexState->GetPointer());
    delete vtk;
    ReplacePressureWithEnergy();
  }

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

  realNeq *pState = vertexState->GetHostPointer();
  for (int n = 0; n < nVertex; n++) {
    pDens[n] = state::GetDensity<realNeq, CL>(pState[n]);
    pMomx[n] = state::GetMomX<realNeq, CL>(pState[n]);
    pMomy[n] = state::GetMomY<realNeq, CL>(pState[n]);
    pEner[n] = state::GetEnergy<realNeq, CL>(pState[n]);
  }

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

  // Check if everything OK
  if (!outFile) {
    std::cout << "Error writing " << fname << std::endl;
    throw std::runtime_error("");
  }

  // Write x-momentum binary
  snprintf(fname, sizeof(fname), "momx%4.4d.dat", nSave);
  outFile.open(fname, std::ios::binary);
  outFile.write(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  outFile.write(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  outFile.write(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  outFile.write(reinterpret_cast<char*>(pMomx), nVertex*sizeof(real));
  outFile.close();

  // Check if everything OK
  if (!outFile) {
    std::cout << "Error writing " << fname << std::endl;
    throw std::runtime_error("");
  }

  // Write y-momentum binary
  snprintf(fname, sizeof(fname), "momy%4.4d.dat", nSave);
  outFile.open(fname, std::ios::binary);
  outFile.write(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  outFile.write(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  outFile.write(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  outFile.write(reinterpret_cast<char*>(pMomy), nVertex*sizeof(real));
  outFile.close();

  // Check if everything OK
  if (!outFile) {
    std::cout << "Error writing " << fname << std::endl;
    throw std::runtime_error("");
  }

  // Write energy binary
  snprintf(fname, sizeof(fname), "ener%4.4d.dat", nSave);
  outFile.open(fname, std::ios::binary);
  outFile.write(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  outFile.write(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  outFile.write(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  outFile.write(reinterpret_cast<char*>(pEner), nVertex*sizeof(real));
  outFile.close();

  // Check if everything OK
  if (!outFile) {
    std::cout << "Error writing " << fname << std::endl;
    throw std::runtime_error("");
  }

  delete dens;
  delete momx;
  delete momy;
  delete ener;

  // Output save number so that we can restore latest save if wanted
  outFile.open("lastsave.dat");
  outFile << nSave << std::endl;
  outFile.close();

  // Check if everything OK
  if (!outFile) {
    std::cout << "Error writing lastsave.dat" << std::endl;
    throw std::runtime_error("");
  }

  std::cout << " Done" << std::endl;

  delete nvtxSave;
}

//#########################################################################
/*! Restore state from previous save.
  \param nRestore Save number to restore.*/
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::Restore(int nRestore)
{
  std::ifstream inFile;
  char fname[13];
  int sizeOfData = sizeof(real);

  if (nRestore == -1) {
    inFile.open("lastsave.dat");
    if (!inFile) {
      std::cout << "Could not open lastsave.dat!"
                << std::endl;
      throw std::runtime_error("");
    }
    inFile >> nSave;
    inFile.close();
  } else {
    nSave = nRestore;
  }

  std::cout << "Restoring save #" << nSave << std::endl;

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
    throw std::runtime_error("");
  }
  inFile.read(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  inFile.read(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  inFile.read(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  inFile.read(reinterpret_cast<char*>(pDens), nVertex*sizeof(real));
  inFile.close();
  if (!inFile) {
    std::cout << "Error reading from " << fname  << ", aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }

  // Read x momentum binary
  snprintf(fname, sizeof(fname), "momx%4.4d.dat", nSave);
  inFile.open(fname, std::ios::binary);
  if (!inFile) {
    std::cout << "Could not open " << fname  << ", aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }
  inFile.read(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  inFile.read(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  inFile.read(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  inFile.read(reinterpret_cast<char*>(pMomx), nVertex*sizeof(real));
  inFile.close();
  if (!inFile) {
    std::cout << "Error reading from " << fname  << ", aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }

  // Read y momentum binary
  snprintf(fname, sizeof(fname), "momy%4.4d.dat", nSave);
  inFile.open(fname, std::ios::binary);
  if (!inFile) {
    std::cout << "Could not open " << fname  << ", aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }
  inFile.read(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  inFile.read(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  inFile.read(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  inFile.read(reinterpret_cast<char*>(pMomy), nVertex*sizeof(real));
  inFile.close();
  if (!inFile) {
    std::cout << "Error reading from " << fname  << ", aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }

  // Read energy binary
  snprintf(fname, sizeof(fname), "ener%4.4d.dat", nSave);
  inFile.open(fname, std::ios::binary);
  if (!inFile) {
    std::cout << "Could not open " << fname  << ", aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }
  inFile.read(reinterpret_cast<char*>(&sizeOfData), sizeof(int));
  inFile.read(reinterpret_cast<char*>(&simulationTime), sizeof(real));
  inFile.read(reinterpret_cast<char*>(&nTimeStep), sizeof(int));
  inFile.read(reinterpret_cast<char*>(pEner), nVertex*sizeof(real));
  inFile.close();
  if (!inFile) {
    std::cout << "Error reading from " << fname  << ", aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }

  realNeq *pState = vertexState->GetHostPointer();
  for (int n = 0; n < nVertex; n++) {
    state::SetDensity<realNeq, CL>(pState[n], pDens[n]);
    state::SetMomX<realNeq, CL>(pState[n], pMomx[n]);
    state::SetMomY<realNeq, CL>(pState[n], pMomy[n]);
    state::SetEnergy<realNeq, CL>(pState[n], pEner[n]);
  }


  // Copy data to device
  if (cudaFlag == 1) vertexState->CopyToDevice();

  // Update simulation.dat
  try {
    RestoreFine();
  }
  catch (...) {
    std::cout << "RestoreFine failed!" << std::endl;
    throw;
  }

  std::cout << "Done restoring " << std::endl;

  nSave++;
}

//#########################################################################
/*! Do a fine grain save, i.e. write output files for certain global quantities but do not do a full data dump.*/
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::FineGrainSave()
{
  Diagnostics<realNeq, CL> *d  =
    new Diagnostics<realNeq, CL>(vertexState, vertexPotential, mesh);
  real *pResult = d->result->GetPointer();

  std::ofstream outFile;

  if (nSave == 0)
    outFile.open("simulation.dat");
  else
    outFile.open("simulation.dat", std::ios::app);

  outFile << std::setprecision(10)
          << simulationTime << " "
          << DensityError() << " ";
  for (int i = 0; i < d->result->GetSize(); i++)
    outFile << pResult[i] << " ";
  outFile << std::endl;

  delete d;
  outFile.close();
  if (!outFile) {
    std::cout << "Error writing simulation.dat!" << std::endl;
    throw std::runtime_error("");
  }
}

//#########################################################################
/*! When restoring a previous dump, we must ensure that we start writing simulation.dat in the correct place. Upon return, the file simulation.dat has been stripped of any excess lines, and nSaveFine is set to the correct number.*/
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::RestoreFine()
{
  // Copy simulation.dat into temp.dat
  std::ifstream src("simulation.dat", std::ios::binary);
  if (!src) {
    std::cout << "Could not open simulation.dat, aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }
  std::ofstream dst("temp.dat", std::ios::binary);
  if (!dst) {
    std::cout << "Could not open temporary file for writing, aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }

  dst << src.rdbuf();

  src.close();
  dst.close();
  if (!dst || !src) {
    std::cout << "Error copying simulation.dat into temp.dat!" << std::endl;
    throw std::runtime_error("");
  }

  // Read from temp.dat and write into simulation.dat
  std::ifstream inFile("temp.dat");
  if (!inFile) {
    std::cout << "Could not open temporary file for reading, aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }
  std::ofstream outFile("simulation.dat");
  if (!outFile) {
    std::cout << "Could not open simulation.dat, aborting restart"
              << std::endl;
    throw std::runtime_error("");
  }

  std::string line;
  while (std::getline(inFile, line)) {
    std::istringstream iss(line);

    // Get time of fine grain save
    real saveTime;
    if (!(iss >> saveTime)) {
      std::cout << "Error reading temporary file, aborting restart" << std::endl;
      throw std::runtime_error("");
    }

    // Write into simulation.dat as long as smaller than current sim time
    if (saveTime <= simulationTime) {
      outFile << line << std::endl;
      nSaveFine++;
    } else {
      break;
    }
  }

  inFile.close();
  outFile.close();
  if (!outFile) {
    std::cout << "Error updating simulation.dat, aborting restart!"
              << " " << !inFile << " " << !outFile << std::endl;
    throw std::runtime_error("");
  }

  // Delete temporary file
  std::remove("temp.dat");
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::Save();
template void Simulation<real, CL_BURGERS>::Save();
template void Simulation<real3, CL_CART_ISO>::Save();
template void Simulation<real4, CL_CART_EULER>::Save();

//##############################################################################

template void Simulation<real, CL_ADVECT>::Restore(int nRestore);
template void Simulation<real, CL_BURGERS>::Restore(int nRestore);
template void Simulation<real3, CL_CART_ISO>::Restore(int nRestore);
template void Simulation<real4, CL_CART_EULER>::Restore(int nRestore);

//##############################################################################

template void Simulation<real, CL_ADVECT>::FineGrainSave();
template void Simulation<real, CL_BURGERS>::FineGrainSave();
template void Simulation<real3, CL_CART_ISO>::FineGrainSave();
template void Simulation<real4, CL_CART_EULER>::FineGrainSave();

//##############################################################################

template void Simulation<real, CL_ADVECT>::RestoreFine();
template void Simulation<real, CL_BURGERS>::RestoreFine();
template void Simulation<real3, CL_CART_ISO>::RestoreFine();
template void Simulation<real4, CL_CART_EULER>::RestoreFine();

}  // namespace astrix
