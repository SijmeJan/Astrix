// -*-c++-*-
/*! \file structured.cpp
\brief Functions for creating structured mesh
*/
#include <iostream>
#include <cmath>
#include <cuda_runtime_api.h>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./Morton/morton.h"
#include "./mesh.h"
#include "./Delaunay/delaunay.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"

namespace astrix {

//#########################################################################
/*! Create a structured mesh of triangles. This is very cheap and has to be done only once (at initialisation) so for now we do it on the host, copying the mesh structure to the CUDA device if necessary. */
//#########################################################################

void Mesh::CreateStructuredMesh()
{
  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;
  
  int nx = (int) (sqrt(0.565/meshParameter->baseResolution)*Px) + 4;
  int ny = (int)(nx*Py/Px);

  if (meshParameter->problemDef == PROBLEM_SOD ||
      meshParameter->problemDef == PROBLEM_BLAST ||
      meshParameter->problemDef == PROBLEM_LINEAR) ny = 2;
  
  std::cout << "Creating structured mesh " << nx << "x" << ny << std::endl;

  // This step is not worth porting to GPU, so do it on the host
  if (cudaFlag == 1) connectivity->Transform();
  
  nVertex = nx*ny;
  nTriangle = (nx - 1)*(ny - 1)*2;
  nEdge = 3*(nx - 1)*(ny - 1) + nx + ny - 2;

  // Allocate memory
  connectivity->vertexCoordinates->SetSize(nVertex);
  connectivity->triangleVertices->SetSize(nTriangle);
  connectivity->triangleEdges->SetSize(nTriangle);
  connectivity->edgeTriangles->SetSize(nEdge);
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer(); 
  
  // Fill vertex coordinates
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      int n = j*nx + i;
      pVc[n].x = Px*i/(real) (nx - 1) + meshParameter->minx;
      pVc[n].y = -Py*j/(real) (ny - 1) + meshParameter->maxy;
    }
  }
  
  // Create mesh
  for (int i = 0; i < nx - 1; i++) {
    for (int j = 0; j < ny - 1; j++) {
      int v = j*nx + i;
      int t = 2*(v - j);
      int e = 3*(v - j) + j;
      
      pTv[t].x = v;
      pTv[t].y = v + nx;
      pTv[t].z = v + 1;

      pTv[t + 1].x = v + 1;
      pTv[t + 1].y = v + nx;
      pTv[t + 1].z = v + 1 + nx;
      
      pTe[t].x = e;
      pTe[t].y = e + 1;
      pTe[t].z = e + 2;

      pTe[t + 1].x = e + 1;
      pTe[t + 1].y = 3*(v + nx - j - 1) + j + 3; //e(j + 1) + 2
      if (j == ny - 2)
	pTe[t + 1].y = 3*(j*nx + nx - 2 + 1 - j) + j + i + 1;
      pTe[t + 1].z = 3*(v + 1 - j) + j; //e(i + 1)

      int e1 = pTe[t].x;
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;
      
      pEt[e1].x = t;
      if (i > 0) pEt[e1].y = t - 1; else pEt[e1].y = -1;
      pEt[e2].x = t;
      pEt[e2].y = t + 1;
      pEt[e3].x = t;
      if (j > 0) pEt[e3].y = 2*(v - nx - j + 1) + 1; else pEt[e3].y = -1;

      e2 = pTe[t + 1].y;
      e3 = pTe[t + 1].z;
      
      if (i == nx - 2) {
	pEt[e3].x = t + 1;
	pEt[e3].y = -1;
      }

      if (j == ny - 2) {
	pEt[e2].x = t + 1;
	pEt[e2].y = -1;
      }
    }
  }
  
  // Make periodic in x
  if (meshParameter->periodicFlagX == 1) {
   for (int i = 0; i < nx; i++) 
      for (int j = 0; j < ny; j++) 
	pVc[j*nx + i].x -= Px*0.5/(real) (nx - 1);
   
    nVertex =
      connectivity->vertexCoordinates->RemoveEvery
      (0, nx, connectivity->triangleVertices);
    nEdge =
      connectivity->edgeTriangles->RemoveEvery
      (0, 3*nx - 2, connectivity->triangleEdges) + 1;
    connectivity->edgeTriangles->SetSize(nEdge);
    
    int i = 0;
    for (int j = 0; j < ny - 1; j++) {
      int v = j*nx + i;
      int t = 2*(v - j);

      pTv[t].x = v - j + (nx - 2);
      pTv[t].y = v - j + nx - 1 + (nx - 2);
      pTv[t].z = v - j + nVertex;
      
      pTv[t + 1].y = v - j + 2*nx - 3 - nVertex;
      
      pTe[t].x = 3*(j*nx + nx - 1 - j) - 1;
      pTe[t].y = pTe[t].z;
      pTe[t].z = pTe[t].y - 1;

      pTe[t + 1].x = pTe[t].y;
      pTe[t + 1].y = pTe[t].x + 1;
      if (j == ny - 2) pTe[t + 1].y = nEdge - 1;
      pTe[t + 1].z = pTe[t + 2].x;

      int e1 = pTe[t].x;
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;
      
      pEt[e1].y = t;
      pEt[e2].x = t;
      pEt[e2].y = t + 1;
      pEt[e3].x = t;
      if (j > 0) pEt[e3].y = 2*((j-1)*nx - j + 1) + 1; else pEt[e3].y = -1;

      if (j == ny - 2) pEt[nEdge - 1].x = t + 1;            
    }
    
    nx--;
  }

  // Make periodic in y
  if (meshParameter->periodicFlagY == 1) {
    for (int i = 0; i < nx; i++) 
      for (int j = 0; j < ny; j++) 
	pVc[j*nx + i].y -= Py*0.5/(real) (ny - 1);
    
    for (int i = 0; i < nTriangle; i++) {
      if (pTv[i].x >= nVertex) pTv[i].x -= nx;
      if (pTv[i].y >= nVertex) pTv[i].y -= nx;
      if (pTv[i].z >= nVertex) pTv[i].z -= nx;

      if (pTv[i].x < 0) pTv[i].x += nx;
      if (pTv[i].y < 0) pTv[i].y += nx;
      if (pTv[i].z < 0) pTv[i].z += nx;
    }
    
    nVertex = nx*(ny - 1);
    connectivity->vertexCoordinates->SetSize(nVertex);

    nEdge -= (nx - 1 + meshParameter->periodicFlagX);
    connectivity->edgeTriangles->SetSize(nEdge);
    
    int j = ny - 2;
    for (int i = 0; i < nx - 1 + meshParameter->periodicFlagX; i++) {
      int v = j*nx + i;
      int t = 2*(v + meshParameter->periodicFlagX);
      if (i == nx - 1) t = 2*j*nx;
      
      pTv[t].y = v - (ny - 2)*nx - 3*nVertex;

      pTv[t + 1].x = v + 1 + 3*nVertex;
      pTv[t + 1].y = v - (ny - 2)*nx;
      pTv[t + 1].z = v + 1 - (ny - 2)*nx;
      if (i == nx - 1) {
	pTv[t + 1].x = j*nx;
	pTv[t + 1].y = nx - 1 - nVertex - 3*nVertex;
	pTv[t + 1].z = -3*nVertex;
      }

      pTe[t + 1].y = 3*i + 2 + 2*meshParameter->periodicFlagX;
      if (i == nx - 1) pTe[t + 1].y = 0;
      
      if (i < nx - 1)
	pEt[3*i + 2 + 2*meshParameter->periodicFlagX].y = t + 1;
      if (i == nx - 1) pEt[pTe[t + 1].y].y = t + 1;
    }
  }

  // Transform back to device
  if (cudaFlag == 1) connectivity->Transform();

  // One cycle of delaunay 
  delaunay->MakeDelaunay(connectivity, 0, predicates,
			 meshParameter, 1, 0, 0, 0);

  morton->Order(connectivity, triangleWantRefine, 0);
  
  // Calculate triangle normals and areas
  CalcNormalEdge();
  CalcVertexArea();
  FindBoundaryVertices();
}

}
