/*! \file linsys.h
\brief Header file for LinSys class*/
#ifndef ASTRIX_LINSYS_H
#define ASTRIX_LINSYS_H

namespace astrix {

class Mesh;
template <class T> class Array;
class Device; 

class LinSys
{
 public:
  //! Constructor for LinSys object.
  LinSys(int _verboseLevel, int _debugLevel, Device *_device);
  //! Destructor, releases all dynamically allocated memory
  ~LinSys();

  void BiCGStab(Array<real> *x, Array<real> *b);
  
 private:
  Device *device;
  
  //! Flag whether to use CUDA
  int cudaFlag;

  //! How much to output to screen
  int verboseLevel;
  //! Level of debugging
  int debugLevel;
  
  //! Mesh on which to do simulation
  Mesh *mesh;

  void MultiplyByMatrix(Array<real> *vIn, Array<real> *vOut);
};

}
#endif  // ASTRIX_LINSYS_H
