/*! \file profile.h
\brief Header file for writing profile info to file.
*/
#ifndef ASTRIX_PROFILE_H
#define ASTRIX_PROFILE_H

namespace astrix{

//! Write kernel profiling info to file
/*! Write simple statistics to text file 
  \param *fileName Output file name
  \param X Number of elements processed
  \param T Elapsed time
*/  
void WriteProfileFile(const char *fileName, int X, float T, int cudaFlag);

}

#endif


