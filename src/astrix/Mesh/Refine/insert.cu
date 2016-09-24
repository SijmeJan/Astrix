// -*-c++-*-
/*! \file insert.cu
\brief Functions to insert new vertices in Mesh*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "../Predicates/predicates.h"
#include "./refine.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {
    
//#########################################################################
/*! \brief Insert vertex \a n into Mesh

\param n Index of vertex to insert
\param t Triangle to insert vertex into (-1 if inserting on edge)
\param e Edge onto which to insert vertex (-1 if inserting in triangle)
\param nVertex Total number of vertices in Mesh
\param nEdge Total number of edges in Mesh
\param nTriangle Total number of triangles in Mesh
\param x x-coordinate of vertex to insert
\param y y-coordinate of vertex to insert
\param indexInEdgeArray Index to start adding edges in \a edgeTriangles
\param indexInTriangleArray Index to start adding triangles in \a triangleVertices and \a triangleEdges
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *te1 Pointer to first edge of triangle 
\param *te2 Pointer to second edge of triangle 
\param *te3 Pointer to third edge of triangle 
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary
\param nvAdd Total number of vertices to be added
\param periodicFlagX Flag whether domain is periodic in x
\param periodicFlagY Flag whether domain is periodic in y
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param *pWantRefine Pointer to array of flags whether to refine triangles. Value of the initial triangle is set to that of the new triangles (either 1 if creating a new Mesh or 0 is refining during a simulation) */
//#########################################################################

__host__ __device__
void InsertVertex(int n, int t, int e, int nVertex, int nEdge,
		  int nTriangle, real x, real y,
		  int indexInEdgeArray, int indexInTriangleArray,
		  real2 *pVc, int3 *pTv, int3 *pTe, int2 *pEt,
		  real minx, real maxx, real miny, real maxy,
		  int nvAdd, int periodicFlagX, int periodicFlagY, 
		  const Predicates *pred, real *pParam,
		  int *pWantRefine)
{
  int i = n + nVertex;  

  // PERIODIC
  int translateVertex = 0;

#ifndef __CUDA_ARCH__
  if (t == -1 && e == -1) {
    std::cout << std::endl
	      << "Error in InsertVertex: no triangle or edge found"
	      << std::endl;
    int qq; std::cin >>qq;
  }
#endif
  
  // Insert vertex inside triangle t
  if (t != -1) {
    if (periodicFlagX) {
      if (x <= minx) {
	translateVertex -= (nVertex + nvAdd);
	x += (maxx - minx);
      }
      if (x > maxx) {
	translateVertex += (nVertex + nvAdd);
	x -= (maxx - minx);
      }
    }
    if (periodicFlagY) {
      if (y <= miny) {
	translateVertex -= 3*(nVertex + nvAdd);
	y += (maxy - miny);
      }
      if (y > maxy) {
	translateVertex += 3*(nVertex + nvAdd);
	y -= (maxy - miny);
      }
    }
    
    pVc[i].x = x;
    pVc[i].y = y;

    if (pWantRefine != 0)
      pWantRefine[t] = pWantRefine[indexInTriangleArray];

    int a = pTv[t].x;
    int b = pTv[t].y;
    int c = pTv[t].z;
    
    // Create triangles
    pTv[indexInTriangleArray].x = c;
    pTv[indexInTriangleArray].y = a;
    pTv[indexInTriangleArray].z = i + translateVertex;
    pTv[indexInTriangleArray + 1].x = b;
    pTv[indexInTriangleArray + 1].y = c;
    pTv[indexInTriangleArray + 1].z = i + translateVertex;
    // Replace triangle t
    pTv[t].z = i + translateVertex;

    MakeValidIndices(pTv[indexInTriangleArray].x,
		     pTv[indexInTriangleArray].y,
		     pTv[indexInTriangleArray].z,
		     nVertex + nvAdd);
    MakeValidIndices(pTv[indexInTriangleArray + 1].x,
		     pTv[indexInTriangleArray + 1].y,
		     pTv[indexInTriangleArray + 1].z,
		     nVertex + nvAdd);
    MakeValidIndices(pTv[t].x, pTv[t].y, pTv[t].z, nVertex + nvAdd);
        
    // Give new edges triangles
    pEt[indexInEdgeArray].x = indexInTriangleArray;
    pEt[indexInEdgeArray].y = t;
    pEt[indexInEdgeArray + 1].x = t;
    pEt[indexInEdgeArray + 1].y = indexInTriangleArray + 1;
    pEt[indexInEdgeArray + 2].x = indexInTriangleArray + 1;
    pEt[indexInEdgeArray + 2].y = indexInTriangleArray;

    int e2 = pTe[t].y;
    int e3 = pTe[t].z;
    
    // Update triangle info for old edges
    if (pEt[e2].x == t) pEt[e2].x = indexInTriangleArray + 1;
    if (pEt[e2].y == t) pEt[e2].y = indexInTriangleArray + 1;
    if (pEt[e3].x == t) pEt[e3].x = indexInTriangleArray;
    if (pEt[e3].y == t) pEt[e3].y = indexInTriangleArray;

    // Give new triangles edges
    pTe[indexInTriangleArray].x = e3;
    pTe[indexInTriangleArray].y = indexInEdgeArray;
    pTe[indexInTriangleArray].z = indexInEdgeArray + 2;
    pTe[indexInTriangleArray + 1].x = e2;
    pTe[indexInTriangleArray + 1].y = indexInEdgeArray + 2;
    pTe[indexInTriangleArray + 1].z = indexInEdgeArray + 1;
    pTe[t].y = indexInEdgeArray + 1;
    pTe[t].z = indexInEdgeArray;

    /*
#ifndef __CUDA_ARCH__
    int T[] = {t, indexInTriangleArray, indexInTriangleArray + 1};

    for (int l = 0; l < 3; l++) {
      int tt = T[l];
      int A = pTv[tt].x;
      int B = pTv[tt].y;
      int C = pTv[tt].z;
    
      real Ax, Bx, Cx, Ay, By, Cy;
      GetTriangleCoordinates(pVc, A, B, C, nVertex + nvAdd,
			     maxx - minx, maxy - miny,
			     Ax, Bx, Cx, Ay, By, Cy);
      
      real area = 0.5*((Ax - Cx)*(By - Cy) - (Ay - Cy)*(Bx - Cx));

      if (area < 0.0) {
	std::cout << "Created triangle with negative area! " << std::endl;
	int qq; std::cin >> qq;
      }
    }
#endif
    */ 
    
  } else {
    // Insert vertex on edge e
    int t1 = pEt[e].x;
    int t2 = pEt[e].y;
    
    if (periodicFlagX) {
      if (x <= minx) translateVertex -= (nVertex + nvAdd);
      if (x > maxx) translateVertex += (nVertex + nvAdd);
    }
    if (periodicFlagY) {
      if (y <= miny) translateVertex -= 3*(nVertex + nvAdd);
      if (y > maxy) translateVertex += 3*(nVertex + nvAdd);
    }
    
    // Check if edge is segment
    if (t1 != -1 && t2 != -1) {
      // Inserting on normal edge (not a segment)
      
      if (pWantRefine != 0) {
	pWantRefine[t1] = pWantRefine[indexInTriangleArray];
	pWantRefine[t2] = pWantRefine[indexInTriangleArray];
      }
      
      int tv11 = pTv[t1].x;
      int tv21 = pTv[t1].y;
      int tv31 = pTv[t1].z;
      int tv12 = pTv[t2].x;
      int tv22 = pTv[t2].y;
      int tv32 = pTv[t2].z;

      int te11 = pTe[t1].x;
      int te21 = pTe[t1].y;
      int te31 = pTe[t1].z;
      int te12 = pTe[t2].x;
      int te22 = pTe[t2].y;
      int te32 = pTe[t2].z;

      // Edge e is between B and C in t1
      int B = tv11;
      int C = tv21;
      if (e == te21) {
	B = tv21;
	C = tv31;
      }
      if (e == te31) {
	B = tv31;
	C = tv11;
      }

      // Edge e is between E and F in t2
      int E = tv12;
      int F = tv22;
      if (e == te22) {
	E = tv22;
	F = tv32;
      }
      if (e == te32) {
	E = tv32;
	F = tv12;
      }

      // Check if one triangle needs translating
      if (B != F || C != E) {
	// Coordinates of t1
	real ax, bx, cx, ay, by, cy;
	GetTriangleCoordinates(pVc, tv11, tv21, tv31,
			       nVertex + nvAdd, maxx - minx, maxy - miny,
			       ax, bx, cx, ay, by, cy);

	// A1: vertex *exactly* on edge of t1?
	real A1 = 0.0;
	if (e == te11) A1 = pred->orient2d(ax, ay, bx, by, x, y, pParam);
	if (e == te21) A1 = pred->orient2d(bx, by, cx, cy, x, y, pParam);
	if (e == te31) A1 = pred->orient2d(cx, cy, ax, ay, x, y, pParam);

	// Coordinates of t2
	GetTriangleCoordinates(pVc, tv12, tv22, tv32,
			       nVertex + nvAdd, maxx - minx, maxy - miny,
			       ax, bx, cx, ay, by, cy);

	// A2: vertex *exactly* on edge of t2?
	real A2 = 0.0;
	if (e == te12) A2 = pred->orient2d(ax, ay, bx, by, x, y, pParam);
	if (e == te22) A2 = pred->orient2d(bx, by, cx, cy, x, y, pParam);
	if (e == te32) A2 = pred->orient2d(cx, cy, ax, ay, x, y, pParam);

	if (A1 == 0.0 && A2 == 0.0) {
	  // Vertex *exactly* on edges of both t1 and t2 
	  real xmax = max(ax, bx);
	  real xmin = min(ax, bx);
	  real ymax = max(ay, by);
	  real ymin = min(ay, by);
	  
	  if (e == te22) {
	    xmax = max(bx, cx);
	    xmin = min(bx, cx);
	    ymax = max(by, cy);
	    ymin = min(by, cy);
	  }
	  
	  if (e == te32) {
	    xmax = max(cx, ax);
	    xmin = min(cx, ax);
	    ymax = max(cy, ay);
	    ymin = min(cy, ay);
	  }

	  int f = (translateVertex == 0);
	  if (x < xmax && x > xmin && y < ymax && y > ymin)
	    f = (translateVertex > 0);

	  if (f == 1) {
	    tv11 -= (B - F);
	    tv21 -= (B - F);
	    tv31 -= (B - F);
	  } else {
	    tv12 += (B - F);
	    tv22 += (B - F);
	    tv32 += (B - F);
	  }	    
	    
	} else {
	  if (fabs(A2) > fabs(A1)) {
	    tv12 += (B - F);
	    tv22 += (B - F);
	    tv32 += (B - F);
	  } else {
	    tv11 -= (B - F);
	    tv21 -= (B - F);
	    tv31 -= (B - F);
	  }
	}
	
	if (B - F != C - E) {
#ifndef __CUDA_ARCH__
	  std::cout << "Not sure what to do inserting on edge " << e
		    << std::endl;
	  std::cout << "t1: " << tv11 << " " << tv21 << " " << tv31
		    << std::endl;
	  std::cout << "t2: " << tv12 << " " << tv22 << " " << tv32
		    << std::endl;
	  std::cout << "Vertex: " << i + translateVertex << std::endl;
	  std::cout << "A1: " << A1 << ", A2: " << A2 << std::endl;
	  int qq; std::cin >> qq;
#endif
	}
      }
      
      if (e == te11) {
	pTv[indexInTriangleArray].x = i + translateVertex;
	pTv[indexInTriangleArray].y = tv21;
	pTv[indexInTriangleArray].z = tv31;
	
	pTv[t1].x = i + translateVertex;
	pTv[t1].y = tv31;
	pTv[t1].z = tv11;
	
	if (pEt[te21].x == t1) pEt[te21].x = indexInTriangleArray;
	if (pEt[te21].y == t1) pEt[te21].y = indexInTriangleArray;

	pTe[indexInTriangleArray].x = indexInEdgeArray;
	pTe[indexInTriangleArray].y = te21;
	pTe[indexInTriangleArray].z = indexInEdgeArray + 1;
	pTe[t1].x = indexInEdgeArray + 1;
	pTe[t1].y = te31;
	pTe[t1].z = e;
      }
      if (e == te21) {
	pTv[indexInTriangleArray].x = i + translateVertex;
	pTv[indexInTriangleArray].y = tv31;
	pTv[indexInTriangleArray].z = tv11;

	pTv[t1].x = i + translateVertex;
	pTv[t1].y = tv11;
	pTv[t1].z = tv21;

      	if (pEt[te31].x == t1) pEt[te31].x = indexInTriangleArray;
	if (pEt[te31].y == t1) pEt[te31].y = indexInTriangleArray;

      	pTe[indexInTriangleArray].x = indexInEdgeArray;
	pTe[indexInTriangleArray].y = te31;
	pTe[indexInTriangleArray].z = indexInEdgeArray + 1;
	pTe[t1].x = indexInEdgeArray + 1;
	pTe[t1].y = te11;
	pTe[t1].z = e;
      }
      if (e == te31) {
	pTv[indexInTriangleArray].x = i + translateVertex;
	pTv[indexInTriangleArray].y = tv11;
	pTv[indexInTriangleArray].z = tv21;

	pTv[t1].x = i + translateVertex;
	pTv[t1].y = tv21;
	pTv[t1].z = tv31;

      	if (pEt[te11].x == t1) pEt[te11].x = indexInTriangleArray;
	if (pEt[te11].y == t1) pEt[te11].y = indexInTriangleArray;
	
	pTe[indexInTriangleArray].x = indexInEdgeArray;
	pTe[indexInTriangleArray].y = te11;
	pTe[indexInTriangleArray].z = indexInEdgeArray + 1;
	pTe[t1].x = indexInEdgeArray + 1;
	pTe[t1].y = te21;
	pTe[t1].z = e;
      }

      if (e == te12) {
	pTv[indexInTriangleArray + 1].x = i + translateVertex;
	pTv[indexInTriangleArray + 1].y = tv32;
	pTv[indexInTriangleArray + 1].z = tv12;

	pTv[t2].x = i + translateVertex;
	pTv[t2].y = tv22;
	pTv[t2].z = tv32;

	if (pEt[te32].x == t2) pEt[te32].x = indexInTriangleArray + 1;
	if (pEt[te32].y == t2) pEt[te32].y = indexInTriangleArray + 1;

	pTe[indexInTriangleArray + 1].x = indexInEdgeArray + 2;
	pTe[indexInTriangleArray + 1].y = te32;
	pTe[indexInTriangleArray + 1].z = indexInEdgeArray;
	pTe[t2].x = e;
	pTe[t2].y = te22;
	pTe[t2].z = indexInEdgeArray + 2;
      }
      if (e == te22) {
	pTv[indexInTriangleArray + 1].x = i + translateVertex;
	pTv[indexInTriangleArray + 1].y = tv12;
	pTv[indexInTriangleArray + 1].z = tv22;

	pTv[t2].x = i + translateVertex;
	pTv[t2].y = tv32;
	pTv[t2].z = tv12;

      	if (pEt[te12].x == t2) pEt[te12].x = indexInTriangleArray + 1;
	if (pEt[te12].y == t2) pEt[te12].y = indexInTriangleArray + 1;

      	pTe[indexInTriangleArray + 1].x = indexInEdgeArray + 2;
	pTe[indexInTriangleArray + 1].y = te12;
	pTe[indexInTriangleArray + 1].z = indexInEdgeArray;
	pTe[t2].x = e;
	pTe[t2].y = te32;
	pTe[t2].z = indexInEdgeArray + 2;
      }
      if (e == te32) {
	pTv[indexInTriangleArray + 1].x = i + translateVertex;
	pTv[indexInTriangleArray + 1].y = tv22;
	pTv[indexInTriangleArray + 1].z = tv32;

	pTv[t2].x = i + translateVertex;
	pTv[t2].y = tv12;
	pTv[t2].z = tv22;
	
      	if (pEt[te22].x == t2) pEt[te22].x = indexInTriangleArray + 1;
	if (pEt[te22].y == t2) pEt[te22].y = indexInTriangleArray + 1;

	pTe[indexInTriangleArray + 1].x = indexInEdgeArray + 2;
	pTe[indexInTriangleArray + 1].y = te22;
	pTe[indexInTriangleArray + 1].z = indexInEdgeArray;
	pTe[t2].x = e;
	pTe[t2].y = te12;
	pTe[t2].z = indexInEdgeArray + 2;
      }

      pEt[indexInEdgeArray].x = indexInTriangleArray;
      pEt[indexInEdgeArray].y = indexInTriangleArray + 1;
      pEt[indexInEdgeArray + 1].x = t1;
      pEt[indexInEdgeArray + 1].y = indexInTriangleArray;
      pEt[indexInEdgeArray + 2].x = indexInTriangleArray + 1;
      pEt[indexInEdgeArray + 2].y = t2;

      // PERIODIC
      MakeValidIndices(pTv[indexInTriangleArray].x,
		       pTv[indexInTriangleArray].y,
		       pTv[indexInTriangleArray].z,
		       nVertex + nvAdd);
      MakeValidIndices(pTv[indexInTriangleArray + 1].x,
		       pTv[indexInTriangleArray + 1].y,
		       pTv[indexInTriangleArray + 1].z,
		       nVertex + nvAdd);
      MakeValidIndices(pTv[t1].x, pTv[t1].y, pTv[t1].z, nVertex + nvAdd);
      MakeValidIndices(pTv[t2].x, pTv[t2].y, pTv[t2].z, nVertex + nvAdd);      
      
      if (periodicFlagX) {
	if (x <= minx) x += (maxx - minx);
	if (x > maxx) x -= (maxx - minx);
      }
      if (periodicFlagY) {
	if (y <= miny) y += (maxy - miny);
	if (y > maxy) y -= (maxy - miny);
      }
      
      pVc[i].x = x;
      pVc[i].y = y;

      /*
#ifndef __CUDA_ARCH__
      int T[] = {t1, t2, indexInTriangleArray, indexInTriangleArray + 1};

      for (int l = 0; l < 4; l++) {
	int tt = T[l];
	int AA = pTv[tt].x;
	int BB = pTv[tt].y;
	int CC = pTv[tt].z;
	
	real Ax, Bx, Cx, Ay, By, Cy;
	GetTriangleCoordinates(pVc, AA, BB, CC, nVertex + nvAdd,
			       maxx - minx, maxy - miny,
			       Ax, Bx, Cx, Ay, By, Cy);
	
	real area = 0.5*((Ax - Cx)*(By - Cy) - (Ay - Cy)*(Bx - Cx));
	
	if (area < 0.0) {
	  std::cout << std::endl
		    << "Created triangle with negative area!"
		    << std::endl
		    << "Inserting vertex at x = " << x << " y = " << y
		    << " on edge " << e << std::endl
		    << "between triangles t1 = " << t1
		    << " with vertices "
		    << pTv[t1].x << " " << pTv[t1].y << " " << pTv[t1].z
		    << std::endl
		    << "and t2 = " << t2
		    << " with vertices "
		    << pTv[t2].x << " " << pTv[t2].y << " " << pTv[t2].z
		    << std::endl
		    << "Translated t1 has vertices: "
		    << tv11 << " " << tv21 << " " << tv31 << std::endl
		    << "Translated t2 has vertices: "
		    << tv12 << " " << tv22 << " " << tv32
		    << std::endl
		    << "Vertex: " << i + translateVertex << std::endl
		    << B << " " << C << " " << E << " " << F << std::endl
		    << A1 << " " << A2 << std::endl;
	  int qq; std::cin >> qq;
	}
      }
#endif
      */
    } else {
      if (periodicFlagX) {
	if (x <= minx) {
	  x += (maxx - minx);
	}
	if (x > maxx) {
	  x -= (maxx - minx);
	}
      }
      if (periodicFlagY) {
	if (y <= miny) {
	  y += (maxy - miny);
	}
	if (y > maxy) {
	  y -= (maxy - miny);
	}
      }
      
      pVc[i].x = x;
      pVc[i].y = y;
      
      // Putting vertex on segment
      int t = t1;
      if (t == -1) t = t2;

      if (pWantRefine != 0)
	pWantRefine[t] = pWantRefine[indexInTriangleArray];

      // Find vertex of neighbouring triangle not belonging to edge
      int v1    = pTv[t].x;
      int vnext = pTv[t].y;
      int vprev = pTv[t].z;

      int E1 = pTe[t].x;
      int E2 = pTe[t].y;
      int E3 = pTe[t].z;
      
      int e1 = E3;
      int e2 = E1;
      if (e == E1) {
	v1    = pTv[t].z;
	vprev = pTv[t].y;
	vnext = pTv[t].x;
	e1 = E2;
	e2 = E3;
      }
      if (e == E3) {
	v1    = pTv[t].y;
	vprev = pTv[t].x;
	vnext = pTv[t].z;
	e1 = E1;
	e2 = E2;
      }

      // PERIODIC
      // Create triangles
      pTv[t].x = i + translateVertex;
      pTv[t].y = vprev;
      pTv[t].z = v1;
      pTv[indexInTriangleArray].x = i + translateVertex;
      pTv[indexInTriangleArray].y = v1;
      pTv[indexInTriangleArray].z = vnext;
      
      // PERIODIC
      MakeValidIndices(pTv[indexInTriangleArray].x,
		       pTv[indexInTriangleArray].y,
		       pTv[indexInTriangleArray].z,
		       nVertex + nvAdd);
      MakeValidIndices(pTv[t].x, pTv[t].y, pTv[t].z, nVertex + nvAdd);
        
      // Give new edges triangles
      pEt[indexInEdgeArray].x = indexInTriangleArray;
      pEt[indexInEdgeArray].y = -1;
      pEt[indexInEdgeArray + 1].x = t;
      pEt[indexInEdgeArray + 1].y = indexInTriangleArray;

      if (pEt[e2].x == t) pEt[e2].x = indexInTriangleArray;
      if (pEt[e2].y == t) pEt[e2].y = indexInTriangleArray;

      // Give new triangles edges
      pTe[indexInTriangleArray].x = indexInEdgeArray + 1;
      pTe[indexInTriangleArray].y = e2;
      pTe[indexInTriangleArray].z = indexInEdgeArray;
      pTe[t].x = e;
      pTe[t].y = e1;
      pTe[t].z = indexInEdgeArray + 1;


      
    }
  }
}

//######################################################################
/*! \brief Insert vertex \a n into Mesh

\param nRefine Total number of vertices to insert
\param *pTriangleAdd Pointer to array of triangle to insert vertex into (-1 if inserting on edge)
\param *pEdgeAdd Pointer to array of edges onto which to insert vertex (-1 if inserting in triangle)
\param nVertex Total number of vertices in Mesh
\param nEdge Total number of edges in Mesh
\param nTriangle Total number of triangles in Mesh
\param *refineX Pointer to array of x-coordinates of vertex to insert
\param *refineY Pointer to array of y-coordinates of vertex to insert
\param *pOnSegmentFlagScan Pointer to array of scanned flags of whether to insert on segment. We need to know this because the number of newly created triangles depends on how many points we insert on segments. 
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *te1 Pointer to first edge of triangle 
\param *te2 Pointer to second edge of triangle 
\param *te3 Pointer to third edge of triangle 
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary
\param nvAdd Total number of vertices to be added
\param periodicFlagX Flag whether domain is periodic in x
\param periodicFlagY Flag whether domain is periodic in y
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param *pWantRefine Pointer to array of flags whether to refine triangles. Value of the initial triangle is set to that of the new triangles (either 1 if creating a new Mesh or 0 is refining during a simulation) */
//######################################################################

__global__ void
devInsertVertices(int nRefine, int *pElementAdd,
                  int nVertex, int nEdge, int nTriangle, real2 *pVcAdd,
		  unsigned int *pOnSegmentFlagScan,
		  real2 *pVc, int3 *pTv, int3 *pTe, int2 *pEt,
		  real minx, real maxx, real miny, real maxy, int nvAdd,
		  int periodicFlagX, int periodicFlagY, 
		  const Predicates *pred, real *pParam,
		  int *pWantRefine)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    int t = pElementAdd[i];
    int e = -1;
    if (t >= nTriangle) {
      e = t - nTriangle;
      t = -1;
    }
  
    InsertVertex(i, t, e,
		 nVertex, nEdge, nTriangle, 
		 pVcAdd[i].x, pVcAdd[i].y,
		 3*i - pOnSegmentFlagScan[i] + nEdge,
		 2*i - pOnSegmentFlagScan[i] + nTriangle,
		 pVc, pTv, pTe, pEt,
		 minx, maxx, miny, maxy,
		 nvAdd, periodicFlagX, periodicFlagY, 
		 pred, pParam, pWantRefine);
    
    i += blockDim.x*gridDim.x;
  }
}
    
//######################################################################
/*! Insert vertices with coordinates as specified in \a vertexCoordinatesAdd into triangles specified in \a triangleAdd or onto edges specified in \a edgeAdd.

  \param *vertexState Pointer to Array containing state vector. We use this to distinguish whether we are creating a new Mesh (\a vertexState=0) meaning all triangles can be refined always, or whether we are in the middle of a simulation in which case no newly created triangles can be refined in the next step before we have computed the new truncation error.*/ 
//######################################################################

void Refine::InsertVertices(Connectivity * const connectivity,
			    const MeshParameter *meshParameter,
			    const Predicates *predicates,
			    Array<realNeq> * const vertexState,
			    Array<int> * const triangleWantRefine)
{
  nvtxEvent *nvtxInsert = new nvtxEvent("Insert", 4);
  nvtxEvent *temp = new nvtxEvent("Segment", 0);

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  int *pElementAdd = elementAdd->GetPointer();
  int nRefine = elementAdd->GetSize();
  
  Array<unsigned int> *onSegmentFlagScan = 
    new Array<unsigned int>(1, cudaFlag, (unsigned int) nRefine);

  int nOnSegment = FlagSegment(connectivity, onSegmentFlagScan);
  unsigned int *pOnSegmentFlagScan = onSegmentFlagScan->GetPointer();
 
  int nv_add = nRefine;
  int ne_add = 3*nv_add - nOnSegment;
  int nt_add = 2*nv_add - nOnSegment;

  delete temp;
  temp = new nvtxEvent("Memory", 1);

  if (verboseLevel > 1) 
    std::cout << "nTriangle: " << nTriangle
	      << ", nEdge: " << nEdge
	      << ", nVertex: " << nVertex 
	      << ", vertices to be inserted in parallel: " << nRefine
	      << " ";

  connectivity->vertexCoordinates->SetSize(nVertex + nv_add);
  connectivity->triangleVertices->SetSize(nTriangle + nt_add);
  connectivity->triangleEdges->SetSize(nTriangle + nt_add);
  connectivity->edgeTriangles->SetSize(nEdge + ne_add);

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();

  // All new edges are flagged to be checked
  edgeNeedsChecking->SetSize(nEdge + ne_add);
  edgeNeedsChecking->SetToSeries(nEdge, nEdge + ne_add);
  
  int *pWantRefine = 0;
  if (triangleWantRefine != 0) {
    triangleWantRefine->SetSize(nTriangle + nt_add);
    triangleWantRefine->SetToValue(1 - (vertexState != 0), nTriangle,
				   triangleWantRefine->GetSize());
    pWantRefine = triangleWantRefine->GetPointer();
  }
  
  real *pParam = predicates->GetParamPointer(cudaFlag);
  
  delete temp;
  temp = new nvtxEvent("Insert", 2);

  // Insert vertices in mesh
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devInsertVertices, 
				       (size_t) 0, 0);

    devInsertVertices<<<nBlocks, nThreads>>>
      (nRefine, pElementAdd,
       nVertex, nEdge, nTriangle, 
       pVcAdd, pOnSegmentFlagScan,
       pVc, pTv, pTe, pEt,
       meshParameter->minx, meshParameter->maxx,
       meshParameter->miny, meshParameter->maxy, nv_add,
       meshParameter->periodicFlagX, meshParameter->periodicFlagY, 
       predicates, pParam, pWantRefine);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRefine; n++) {      
      int t = pElementAdd[n];
      int e = -1;
      if (t >= nTriangle) {
	e = t - nTriangle;
	t = -1;
      }
      
      InsertVertex(n, t, e, nVertex, nEdge, nTriangle,
		   pVcAdd[n].x, pVcAdd[n].y,
		   3*n - pOnSegmentFlagScan[n] + nEdge,
		   2*n - pOnSegmentFlagScan[n]+ nTriangle,
		   pVc, pTv, pTe, pEt,
		   meshParameter->minx, meshParameter->maxx,
		   meshParameter->miny, meshParameter->maxy, nv_add,
		   meshParameter->periodicFlagX, meshParameter->periodicFlagY,
		   predicates, pParam, pWantRefine);
    }
  }
  
  // Update number of vertices, triangles and edges
  //nVertex   += nv_add;
  //nTriangle += nt_add;
  //nEdge     += ne_add;
  
  delete onSegmentFlagScan;

  delete temp;
  delete nvtxInsert;
}

}
