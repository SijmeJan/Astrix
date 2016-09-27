/*! \file simulation.h
\brief Header file for Simulation class*/
#ifndef ASTRIX_SIMULATION_H
#define ASTRIX_SIMULATION_H

namespace astrix {

class Mesh;
template <class T> class Array;
class Device; 

//! Simulation: class containing simulation
/*! This is the basic class needed to run an Astrix simulation.  */

class Simulation
{
 public:
  //! Constructor for Simulation object.
  Simulation(int _verboseLevel, int _debugLevel,
	     char *fileName, Device *_device, int restartNumber,
	     int extraFlag);
  //! Destructor, releases all dynamically allocated 
  ~Simulation();

  //! Run simulation
  void Run(int restartNumber, real maxWallClockHours);
  
 private:
  Device *device;
  
  //! Flag whether to use CUDA
  int cudaFlag;

  //! How much to output to screen
  int verboseLevel;
  //! Level of debugging
  int debugLevel;
  int extraFlag;
  
  //! Mesh on which to do simulation
  Mesh *mesh;

  //! Number of time steps taken
  int nTimeStep;
  
  //! Problem specification (see Common/definitions.h)
  /*! Read from input file: LIN: Linear wave, 
      RT: Rayleigh-Taylor, 
      KH: Kelvin-Helmholz, 
      RIEMANN: 2D Riemann problem, 
      SOD: Sod shock tube, 
      VORTEX: Vortex advection, 
      Converted to int using definitions in definitions.h.*/
  ProblemDefinition problemDef;

  //! Integration scheme (see Common/definitions.h)
  /*! Read from input file: N: N scheme, LDA: LDA scheme, B: blended scheme.*/
  IntegrationScheme intScheme;
  //! Order of accuracy in time (1 or 2)
  int integrationOrder;
  //! Mass matrix formulation to use (1, 2, 3 or 4)
  int massMatrix;
  //! Flag whether to use selective lumping
  int selectiveLumpFlag;
  //! Courant number
  real CFLnumber;
  
  //! Number of space dimensions (fixed to 2)
  int nSpaceDim;

  //! Ratio of specific heats
  real specificHeatRatio;
  
  //! Time variable.
  real simulationTime;
  //! Maximum simulation time 
  real maxSimulationTime;
  //! Time between 2D saves
  real saveIntervalTime;
  //! Time between 0D saves
  real saveIntervalTimeFine;
  
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
  //! Blend factor
  Array <realNeq> *triangleBlendFactor;
  //! Shock sensor
  Array<real> *triangleShockSensor;

  //! Set up the simulation
  void Init(int restartNumber);
  //! Read input file
  void ReadInputFile(const char *fileName);
  
  //! Save current state
  void Save(int nSave);
  //! Restore state from disc
  int Restore(int nSave);
  void FineGrainSave(int nSave);
  void KHDiagnostics(real& M, real& Ekin);

  //! Do one time step
  void DoTimeStep();

  //! Set initial conditions according to problemSpec.
  void SetInitial();

  //! Calculate gravitational potential
  void CalcPotential();
  
  //! For every vertex, calculate the maximum allowed timestep.  
  real CalcVertexTimeStep();
  //! Set reflecting boundary conditions
  void ReflectingBoundaries(real dt);
  //! Set boundary conditions for 2D Riemann problem
  void SetRiemannBoundaries();
  //! Set boundary conditions using extrapolation	
  void ExtrapolateBoundaries();
  //! Set non-reflecting boundaries
  void SetNonReflectingBoundaries();

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
  //! Calculate blend factor LDA/N
  void CalcBlend();
  //! Find unphysical state and put in vertexUnphysicalFlag
  void FlagUnphysical(Array<int> *vertexUnphysicalFlag);
  //! Replace LDA with N wherever unphysical state
  void ReplaceLDA(Array<int> *vertexUnphysicalFlag, int RKStep);

  //! Refine mesh
  void Refine();
  //! Coarsen mesh
  void Coarsen(int maxCycle);
  
  //! In state vector, replace total energy with pressure
  void ReplaceEnergyWithPressure();
  //! In state vector, replace pressure with total energy
  void ReplacePressureWithEnergy();

  real TotalMass();
  //void CalcVorticity(Array<real> *vorticity);

  void CalcShockSensor();
  void CheckSymmetry();

  void FindMinMaxVelocity(real& minVel, real& maxVel);

  void CalcTotalSpaceResidual();
  void UpdateFirst(real dt);
};

}
#endif  // ASTRIX_SIMULATION_H
