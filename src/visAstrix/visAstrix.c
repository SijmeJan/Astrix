#include "visAstrix.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// window identifier
int win;

int winSizeX, winSizeY;

float glut_minx, glut_maxx, glut_miny, glut_maxy;
float glut_current_minx, glut_current_maxx,
  glut_current_miny, glut_current_maxy;

// Vertex coordinates
float *vertX, *vertY;
// Triangle and edge vertices
int *triVert, *triEdge;

// Number of vertices, triangles, edges
int n_vertex, n_triangle, n_edge;
int ndim;
int nSave = 0;
int nStart = 0;

// State at vertices
float *vertDens, *vertVelx, *vertVely, *vertPres;
float *triBlend;

float minx = 0.0, maxx = 1.0, miny = 0.0, maxy = 1.0;

float mindens=1.0e10, maxdens=-1.0e10;
float minvelx=1.0e10, maxvelx=-1.0e10;
float minvely=1.0e10, maxvely=-1.0e10;
float minpres=1.0e10, maxpres=-1.0e10;
float minblend=1.0e10, maxblend=-1.0e10;

int display_grid = 0;
int display_triangle_numbers = 0;
int evolve_flag = 0;
int boundary_flag = 0;
int highlight_flag = 0;
int vector_flag = 0;
float zoom_factor = 1.0;
int plot_var = 0;
int movie_flag = 0;
int copy_flag = 0;

//###########################################################################
// main
//###########################################################################

int main(int argc, char *argv[])
{
  int i, nSwitches = 0;

  // Parse command line arguments
  for (i = 1; i < argc; ++i) {
    // Check if saving movie
    if (strcmp(argv[i],"--movie")==0 || 
	strcmp(argv[i],"-m")==0){
      printf("Saving movie frames\n");
      system("rm -f -r movie_frames");
      system("mkdir movie_frames");
      movie_flag=1;
      nSwitches++;
    }
    // Check if not starting at t=0
    if (strcmp(argv[i],"--number")==0 || 
	strcmp(argv[i],"-n")==0){
      if(i == argc - 1){
	printf("Number missing after -n\n");
	return 1;
      }
      printf("Starting from n = %s\n", argv[i+1]);
      nSave = atoi(argv[i+1]);
      nStart = nSave;
      nSwitches += 2;
    }

  }
  // Check for correct number of arguments
  if (argc != 2 + nSwitches) {
    printf("Usage: %s [-m] [-n] startNumber filename\n", argv[0]); 
    return 1;
  }

  // Last argument should be input file name
  char *fileName = argv[argc-1];

  char line[80];

  FILE *fr = fopen (fileName, "rt");  
  if (fr == NULL) {
    printf("Error opening file %s\n", fileName);
    return 1;
  }

  // Read domain boundaries from input file
  while(fgets(line, 80, fr) != NULL) {
    int j = 0;
    while(line[j] != ' ' && j < 80) j++;
    while((line[j] == ' ' || line[j] == '\t') && j < 80) j++;

    char *token = strtok(line, " ");
    if (strstr (token, "minX") != NULL) 
      minx = atof(strtok(&(line[j]), " "));
    if (strstr (token, "maxX") != NULL) 
      maxx = atof(strtok(&(line[j]), " "));
    if (strstr (token, "minY") != NULL) 
      miny = atof(strtok(&(line[j]), " "));
    if (strstr (token, "maxY") != NULL) 
      maxy = atof(strtok(&(line[j]), " "));
  }
  fclose(fr); 

  if(ReadFiles(1)) return 1;
  
  glutInit(&argc, argv);

  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

  // Calculate window size
  for (i = 0; i < n_vertex; i++) {
    if(vertX[i] > glut_maxx) glut_maxx = vertX[i];
    if(vertX[i] < glut_minx) glut_minx = vertX[i];
    if(vertY[i] > glut_maxy) glut_maxy = vertY[i];
    if(vertY[i] < glut_miny) glut_miny = vertY[i];
  }

  int wideFlag = 1;
  float f = 0.1;
  if (wideFlag == 1) f = 2.5;

  glut_minx = minx - f*(maxx - minx);
  glut_maxx = maxx + f*(maxx - minx);
  glut_miny = miny - f*(maxy - miny);
  glut_maxy = maxy + f*(maxy - miny);
  
  glut_current_minx = glut_minx;
  glut_current_maxx = glut_maxx;
  glut_current_miny = glut_miny;
  glut_current_maxy = glut_maxy;

  float sizeratio = (glut_maxy-glut_miny)/(glut_maxx-glut_minx);
  if(sizeratio > 1.0) sizeratio = 1.0/sizeratio;
  int maxWinSize=768;
  if((glut_maxy-glut_miny) < (glut_maxx-glut_minx)){
    winSizeX = maxWinSize;
    winSizeY = (int)(sizeratio*(float)maxWinSize);
  }else{
    winSizeY = maxWinSize;
    winSizeX = (int)(sizeratio*(float)maxWinSize);
  }

  while(winSizeX % 4 != 0) winSizeX++;
  while(winSizeY % 4 != 0) winSizeY++;

  if (winSizeX < 32) winSizeX = 32;
  if (winSizeY < 32) winSizeY = 32;
  
  printf("Window size: %d %d\n", winSizeX, winSizeY);

  glutInitWindowSize(winSizeX,winSizeY);
  glutInitWindowPosition(500,100);
  
  win = glutCreateWindow("Astrix");

  char winTitle[20];
  sprintf(winTitle, "Astrix frame %3.3d", nSave-1);
  glutSetWindowTitle(winTitle);

  glutDisplayFunc(dispAstrix);
  glutKeyboardFunc(keybAstrix);
  glutIdleFunc(dispAstrix);
  glutReshapeFunc(resizeAstrix);

  // define the color we use to clearscreen 
  glClearColor(0.0,0.0,0.0,0.0);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(glut_minx, glut_maxx,
	  glut_miny, glut_maxy, -1.0, 1.0);  

  // Set data alignment (for taking screenshots)
  glPixelStorei(GL_PACK_ALIGNMENT, 1);

  // enter the main loop
  glutMainLoop();

  return 0;    
}
