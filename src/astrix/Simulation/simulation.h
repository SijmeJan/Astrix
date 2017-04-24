/*! \file simulation.h
\brief Header file for Simulation class

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#ifndef ASTRIX_SIMULATION_H
#define ASTRIX_SIMULATION_H

#define BURGERS -1

namespace astrix {

class Mesh;
template <class T> class Array;
class Device;
class SimulationParameter;

//! Simulation: class containing simulation
/*! This is the basic class needed to run an Astrix simulation.  */

class Simulation
{
 public:
  //! Constructor for Simulation object.
  Simulation(int _verboseLevel, int _debugLevel,
             char *fileName, Device *_device, int restartNumber);
  //! Destructor, releases all dynamically allocated memory
  ~Simulation();

  //! Run simulation
  void Run(real maxWallClockHours);

 private:
  //! GPU device available
  Device *device;
  //! Class holding parameters for the simulation
  SimulationParameter *simulationParameter;

  //! Flag whether to use CUDA
  int cudaFlag;
  //! How much to output to screen
  int verboseLevel;
  //! Level of debugging
  int debugLevel;

  //! Mesh on which to do simulation
  Mesh *mesh;

  //! Number of time steps taken
  int nTimeStep;

  //! Time variable.
  real simulationTime;
  //! Number of saves so far
  int nSave;
  //! Number of fine grain saves so far
  int nSaveFine;

  //! State vector at vertex
  Array <realNeq> *vertexState;
  //! Old state vector at vertex
  Array <realNeq> *vertexStateOld;
  //! Gravitational potential at vertex
  Array <real> *vertexPotential;
  //! Difference state vector at vertex
  Array <realNeq> *vertexStateDiff;
  //! Roe parameter vector
  Array <realNeq> *vertexParameterVector;

  //! Residual for N scheme
  Array <realNeq> *triangleResidueN;
  //! Residual for LDA scheme
  Array <realNeq> *triangleResidueLDA;
  //! Total residual
  Array <realNeq> *triangleResidueTotal;
  //! Shock sensor
  Array<real> *triangleShockSensor;
  //! Source contribution to residual
  Array <realNeq> *triangleResidueSource;

  //! Set up the simulation
  void Init(int restartNumber);

  //! Save current state
  void Save();
  //! Restore state from disc
  void Restore(int nRestore);
  //! Fine grain save
  void FineGrainSave();
  //! Make fine grain save file consistent when restoring
  void RestoreFine();
 //! Calculate Kelvin-Helmholtz diagnostics
  void KHDiagnostics(real& M, real& Ekin);
  //! Add eigenvector perturbation for KH problem
  void KHAddEigenVector();
  //! Add eigenvector perturbation for RT problem
  void RTAddEigenVector();

  //! Do one time step
  void DoTimeStep();

  //! Set initial conditions according to problemSpec.
  void SetInitial(real time);

  //! Calculate gravitational potential
  void CalcPotential();
  //! Calculate source term contribution to residual
  void CalcSource(Array<realNeq> *state);
  //! For every vertex, calculate the maximum allowed timestep.
  real CalcVertexTimeStep();

  //! Set reflecting boundary conditions
  void ReflectingBoundaries(real dt);
  //! Set boundary conditions using extrapolation
  void ExtrapolateBoundaries();
  //! Set non-reflecting boundaries
  void SetNonReflectingBoundaries();
  //! Set boundary conditions for 2D Riemann problem
  void SetRiemannBoundaries();
  //! Set boundary conditions for 2D Noh problem
  void SetNohBoundaries();
  //! Set symmetry boundary conditions
  void SetSymmetricBoundaries();

  //! Function to calculate Roe's parameter vector at all vertices.
  void CalculateParameterVector(int useOldFlag);
  //! Calculate space residual on triangles
  void CalcResidual();
  //! Calculate space-time residual N plus total
  void CalcTotalResNtot(real dt);
  //! Calculate space-time LDA residual
  void CalcTotalResLDA();
  //! Add selective lump contribution to residual
  void SelectLump(real dt, int massMatrix, int selectLumpFlag);
  //! Add contribution F3/F4 mass matrix to total residual
  void MassMatrixF34Tot(real dt, int massMatrix);
  //! Add contribution F3/F4 mass matrix to residual
  void MassMatrixF34(real dt, int massMatrix);

  //! Update state at nodes
  void UpdateState(real dt, int RKStep);
  //! Add residue to state at vertices
  void AddResidue(real dt);
  //! Find unphysical state and put in vertexUnphysicalFlag
  void FlagUnphysical(Array<int> *vertexUnphysicalFlag);
  //! Find changes that are too large
  void FlagLimit(Array<int> *vertexLimitFlag);
  //! Replace LDA with N wherever unphysical state
  void ReplaceLDA(Array<int> *vertexUnphysicalFlag, int RKStep);
  //! Calculate shock sensor for BX scheme
  void CalcShockSensor();
  //! Find minimum and maximum velocity in domain
  void FindMinMaxVelocity(real& minVel, real& maxVel);

  //! Refine mesh
  void Refine();
  //! Coarsen mesh
  void Coarsen(int maxCycle);

  //! In state vector, replace total energy with pressure
  void ReplaceEnergyWithPressure();
  //! In state vector, replace pressure with total energy
  void ReplacePressureWithEnergy();

  //! Calculate total mass in domain
  real TotalMass();
  //! Calculate total kinetic energy in domain
  real2 KineticEnergy();
  //! Calculate total thermal energy in domain
  real ThermalEnergy();
  //! Calculate total potential energy in domain
  real PotentialEnergy();
  //! Calculate L1 density error
  real DensityError();
};

}  // namespace astrix
#endif  // ASTRIX_SIMULATION_H
