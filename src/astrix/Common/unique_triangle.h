/*! \file unique_triangle.h 
\brief Header for finding unique triangles.
*/
#ifndef ASTRIX_UNIQUE_TRIANGLE
#define ASTRIX_UNIQUE_TRIANGLE

namespace astrix {

void FindUniqueTriangleAffected(Array<int> *triangleAffected,
				Array<int> *triangleAffectedIndex,
				Array<int> *uniqueFlag,
				int maxIndex, int cudaFlag);

}

#endif
