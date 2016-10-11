#include "visAstrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

extern int nSave, nStart;

// Vertex coordinates
extern float *vertX, *vertY;
// Triangle vertices
extern int *triVert, *triEdge;

// Number of vertices, triangles, edges
extern int n_vertex, n_triangle, n_edge;
extern int ndim;

// State at vertices
extern float *vertDens, *vertVelx, *vertVely, *vertPres;
extern float *triBlend;

extern float maxx, minx, maxy, miny;

extern float mindens, maxdens;
extern float minvelx, maxvelx;
extern float minvely, maxvely;
extern float minpres, maxpres;
extern float minblend, maxblend;

//###########################################################################
// main
//###########################################################################

int ReadFiles(int startFlag)
{
  FILE *fp=NULL;
  int i;
  char fname[13];

  snprintf(fname, sizeof(fname), "vert%4.4d.dat", nSave);
  fp = fopen(fname, "rb");

  if (!fp) return 0;

  // Number of dimensions
  fread(&ndim, sizeof(int), 1, fp);
  printf("Number of dimensions: %d\n", ndim);
  
  // Size of floating point numbers (float or double)
  int sizeOfData;
  fread((char*) &sizeOfData, sizeof(int), 1, fp);
  printf("Size of data: %d\n", sizeOfData);
  if(sizeOfData != sizeof(float) && sizeOfData != sizeof(double)){
    printf("Unrecognised data size (must be float or double)\n");
    return 1;
  }
  
  // Number of vertices
  fread((char*) &n_vertex, sizeof(int), 1, fp);
  printf("Number of vertices: %d\n", n_vertex);
  
  if (startFlag != 1) {
    free(vertX);
    free(vertY);
  }
  vertX = malloc(n_vertex*sizeof(float));
  vertY = malloc(n_vertex*sizeof(float));
  
  if(sizeOfData == sizeof(double)){
    double *tempX = malloc(n_vertex*sizeof(double));
    double *tempY = malloc(n_vertex*sizeof(double));
    
    fread((char*) tempX, sizeof(double), n_vertex, fp);
    fread((char*) tempY, sizeof(double), n_vertex, fp);
    
    for(i=0;i<n_vertex;i++){
      vertX[i] = (float) tempX[i];
      vertY[i] = (float) tempY[i];
    }
    
    free(tempX);
    free(tempY);    
  }else{
    fread((char*) vertX, sizeof(float), n_vertex, fp);
    fread((char*) vertY, sizeof(float), n_vertex, fp);
  }
  
  fclose(fp);

  snprintf(fname, sizeof(fname), "tria%4.4d.dat", nSave);
  fp = fopen(fname, "rb");
  if (!fp) return 0;

  // Read number of triangles
  fread(&n_triangle, sizeof(int), 1, fp);
  if (startFlag != 1) {
    free(triVert);
    free(triEdge);
  }
  triVert = malloc(n_triangle*(ndim+1)*sizeof(int));
  triEdge = malloc(n_triangle*(ndim+1)*sizeof(int));
  // Read triangle data
  fread((char*) triVert, sizeof(int), (ndim+1)*n_triangle, fp);
  fread((char*) triEdge, sizeof(int), (ndim+1)*n_triangle, fp);
  fclose(fp);
    
  printf("Mesh data read\n");

  if (startFlag != 1) {
    free(vertDens);
    free(vertVelx);
    free(vertVely);
    free(vertPres);
  }

  vertDens = malloc(n_vertex*sizeof(float));
  vertVelx = malloc(n_vertex*sizeof(float));
  vertVely = malloc(n_vertex*sizeof(float));
  vertPres = malloc(n_vertex*sizeof(float));
  
  // Read density binary
  sprintf(fname, "dens%4.4d.dat", nSave);
  fp = fopen(fname, "rb");

  // No file found, return    
  if (!fp) {
    //if (startFlag == 1) {
      for (i=0; i < n_vertex; i++) vertDens[i] = 0.0;
      nSave++;
      return 0;
      //} else {
      //  return 1;
      //}
  }

  float simulationTime = 0.0;
  int nTimeStep = 0;
  
  // Size of floating point numbers (float or double)
  fread((char*) &sizeOfData, sizeof(int), 1, fp);

  if(sizeOfData != sizeof(float) && sizeOfData != sizeof(double)){
    printf("Unrecognised data size (must be float or double)\n");
    return 1;
  }


  if(sizeOfData == sizeof(double)){
    double tempTime;
    fread((char*) &tempTime, sizeOfData, 1, fp);
    simulationTime = (float) tempTime;

    fread((char*) &nTimeStep, sizeof(int), 1, fp);

    double *temp = malloc(n_vertex*sizeof(double));
    fread((char*) temp, sizeof(double), n_vertex, fp);

    for(i=0;i<n_vertex;i++)
      vertDens[i] = (float) temp[i];

    free(temp);
  }else{
    fread((char*) &simulationTime, sizeOfData, 1, fp);
    fread((char*) &nTimeStep, sizeof(int), 1, fp);
    fread((char*) vertDens, sizeof(float), n_vertex, fp);
  }

  fclose(fp);

  // Read momx binary
  sprintf(fname, "momx%4.4d.dat", nSave);
  fp = fopen(fname, "rb");

  // No file found, return    
  if (!fp) {
    if (startFlag == 1) {
      for (i=0; i < n_vertex; i++) vertVelx[i] = 0.0;
      return 0;
    } else {
      return 1;
    }
  }

  // Size of floating point numbers (float or double)
  fread((char*) &sizeOfData, sizeof(int), 1, fp);
  if(sizeOfData != sizeof(float) && sizeOfData != sizeof(double)){
    printf("Unrecognised data size (must be float or double)\n");
    return 1;
  }

  if(sizeOfData == sizeof(double)){
    double tempTime;
    fread((char*) &tempTime, sizeOfData, 1, fp);
    simulationTime = (float) tempTime;
    
    fread((char*) &nTimeStep, sizeof(int), 1, fp);

    double *temp = malloc(n_vertex*sizeof(double));
    fread((char*) temp, sizeof(double), n_vertex, fp);

    for(i=0;i<n_vertex;i++)
      vertVelx[i] = (float) temp[i];

    free(temp);
  }else{
    fread((char*) &simulationTime, sizeOfData, 1, fp);
    fread((char*) &nTimeStep, sizeof(int), 1, fp);
    fread((char*) vertVelx, sizeof(float), n_vertex, fp);
  }

  fclose(fp);

  // Read momy binary
  sprintf(fname, "momy%4.4d.dat", nSave);
  fp = fopen(fname, "rb");

  // No file found, return    
  if (!fp) {
    if (startFlag == 1) {
      for (i=0; i < n_vertex; i++) vertVely[i] = 0.0;
      return 0;
    } else {
      return 1;
    }
  }

  // Size of floating point numbers (float or double)
  fread((char*) &sizeOfData, sizeof(int), 1, fp);
  if(sizeOfData != sizeof(float) && sizeOfData != sizeof(double)){
    printf("Unrecognised data size (must be float or double)\n");
    return 1;
  }

  if(sizeOfData == sizeof(double)){
    double tempTime;
    fread((char*) &tempTime, sizeOfData, 1, fp);
    simulationTime = (float) tempTime;
    
    fread((char*) &nTimeStep, sizeof(int), 1, fp);

    double *temp = malloc(n_vertex*sizeof(double));
    fread((char*) temp, sizeof(double), n_vertex, fp);

    for(i=0;i<n_vertex;i++)
      vertVely[i] = (float) temp[i];

    free(temp);
  }else{
    fread((char*) &simulationTime, sizeOfData, 1, fp);
    fread((char*) &nTimeStep, sizeof(int), 1, fp);
    fread((char*) vertVely, sizeof(float), n_vertex, fp);
  }

  fclose(fp);

  // Read energy binary
  sprintf(fname, "ener%4.4d.dat", nSave);
  fp = fopen(fname, "rb");

  // No file found, return    
  if (!fp) {
    if (startFlag == 1) {
      for (i=0; i < n_vertex; i++) vertPres[i] = 0.0;
      return 0;
    } else {
      return 1;
    }
  }

  // Size of floating point numbers (float or double)
  fread((char*) &sizeOfData, sizeof(int), 1, fp);
  if(sizeOfData != sizeof(float) && sizeOfData != sizeof(double)){
    printf("Unrecognised data size (must be float or double)\n");
    return 1;
  }

  if(sizeOfData == sizeof(double)){
    double tempTime;
    fread((char*) &tempTime, sizeOfData, 1, fp);
    simulationTime = (float) tempTime;
    
    fread((char*) &nTimeStep, sizeof(int), 1, fp);

    double *temp = malloc(n_vertex*sizeof(double));
    fread((char*) temp, sizeof(double), n_vertex, fp);

    for(i=0;i<n_vertex;i++)
      vertPres[i] = (float) temp[i];

    free(temp);
  }else{
    fread((char*) &simulationTime, sizeOfData, 1, fp);
    fread((char*) &nTimeStep, sizeof(int), 1, fp);
    fread((char*) vertPres, sizeof(float), n_vertex, fp);
  }

  fclose(fp);

  for (i = 0; i < n_vertex; i++) {
    vertPres[i] =
      0.4*(vertPres[i] -
	   0.5f*(vertVelx[i]*vertVelx[i] +
		 vertVely[i]*vertVely[i])/vertDens[i]);

    vertVelx[i] = vertVelx[i]/vertDens[i];
    vertVely[i] = vertVely[i]/vertDens[i];
  }

  triBlend = malloc(n_triangle*sizeof(float));
  for (i=0; i < n_triangle; i++) triBlend[i] = 0.0;
  
  // Read blend binary
  sprintf(fname, "blnd%4.4d.dat", nSave);
  fp = fopen(fname, "rb");

  if (fp) {
    /*
  // No file found, return    
  if (!fp) {
    if (startFlag == 1) {
      for (i=0; i < n_triangle; i++) triBlend[i] = 0.0;
      return 0;
    } else {
      return 1;
    }
  }
    */
  // Size of floating point numbers (float or double)
  fread((char*) &sizeOfData, sizeof(int), 1, fp);
  if(sizeOfData != sizeof(float) && sizeOfData != sizeof(double)){
    printf("Unrecognised data size (must be float or double)\n");
    return 1;
  }

  triBlend = malloc(n_triangle*sizeof(float));

  if(sizeOfData == sizeof(double)){
    double tempTime;
    fread((char*) &tempTime, sizeOfData, 1, fp);
    simulationTime = (float) tempTime;
    
    fread((char*) &nTimeStep, sizeof(int), 1, fp);

    double *temp = malloc(n_triangle*sizeof(double));
    fread((char*) temp, sizeof(double), n_triangle, fp);

    for(i=0;i<n_triangle;i++)
      triBlend[i] = (float) temp[i];

    free(temp);
  }else{
    fread((char*) &simulationTime, sizeOfData, 1, fp);
    fread((char*) &nTimeStep, sizeof(int), 1, fp);
    fread((char*) triBlend, sizeof(float), n_triangle, fp);
  }

  //for (i = 0;i < n_triangle; i++)
  //  triBlend[i] = log10(triBlend[i]);

  fclose(fp);
  }
  
  if (nSave == nStart) {
    printf("VertXY: %f %f\n", vertX[0], vertY[0]);

    mindens = 1.0e10;
    maxdens = -1.0e10;
    minvelx = 1.0e10;
    maxvelx = -1.0e10;
    minvely = 1.0e10;
    maxvely = -1.0e10;
    minpres = 1.0e10;
    maxpres = -1.0e10;
    minblend = 1.0e10;
    maxblend = -1.0e10;

    for (i = 0; i < n_vertex; i++) {
      float d = vertDens[i];
      if(d < mindens) mindens=d;
      if(d > maxdens) maxdens=d;
      d = vertVelx[i];
      if(d < minvelx) minvelx=d;
      if(d > maxvelx) maxvelx=d;
      d = vertVely[i];
      if(d < minvely) minvely=d;
      if(d > maxvely) maxvely=d;
      d = vertPres[i];
      if(d < minpres) minpres=d;
      if(d > maxpres) maxpres=d;
    }
    for (i = 0; i < n_triangle; i++) {
      float d = triBlend[i];
      if (d < minblend) minblend = d;
      if (d > maxblend) maxblend = d;

      //printf("blend: %d %e\n", i, d);
    }
    printf("MinMaxBlend: %e %e\n", minblend, maxblend);
    //printf("MinMaxPres: %e %e\n", minpres, maxpres);
    //printf("MinMaxVelX: %e %e\n", minvelx, maxvelx);
    //printf("MinMaxVelY: %e %e\n", minvely, maxvely);
    printf("MinMaxDens: %e %e\n", mindens, maxdens);
   }

  nSave++;
  
  return 0;
}

 
