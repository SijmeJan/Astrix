#include "visAstrix.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

extern int winSizeX, winSizeY;

extern float glut_minx, glut_maxx, glut_miny, glut_maxy;
extern float glut_current_minx, glut_current_maxx, glut_current_miny, glut_current_maxy;
extern float maxx, minx, maxy, miny;


// Vertex coordinates
extern float *vertX, *vertY;
// Triangle vertices
extern int *triVert;//, *edgeVert, *edgeSegment;

// Number of vertices, triangles, edges
extern int n_vertex, n_triangle, n_edge;
extern int ndim;

extern int nSave;

// State at vertices
extern float *vertDens, *vertVelx, *vertVely, *vertPres;

extern int display_grid;
extern int evolve_flag;
extern int boundary_flag;
extern int highlight_flag;
extern int vector_flag;
extern float zoom_factor;
extern int plot_var;
extern int display_triangle_numbers;
extern int copy_flag;

extern
void GetTriangleCoordinates(int i, float *pVertX, float *pVertY,
			    int *tv1, int *tv2, int *tv3, int nVertex,
			    float minx, float maxx, float miny, float maxy,
			    float *Ax, float *Bx, float *Cx,
			    float *Ay, float *By, float *Cy);

int tZoom = -1;
int vZoom = -1;

//###########################################################################
// 
//###########################################################################

void keybAstrix(unsigned char key, int x, int y){
  if(key == 'r'){
    free(vertX);
    free(vertY);
    free(triVert);
    free(vertDens);
    free(vertVelx);
    free(vertVely);
    free(vertPres);

    nSave = 0;

    ReadFiles(1);
  }
  if (key == '+') {
    ReadFiles(0);

    char winTitle[25];
    sprintf(winTitle, "Astrix frame %3.3d", nSave-1);
    glutSetWindowTitle(winTitle);
  }
  if (key == '-') {
    if (nSave > 1) {
      nSave-=2;
      ReadFiles(0);
      
      printf("Previous frame: %d\n", nSave);
      
      char winTitle[25];
      sprintf(winTitle, "Astrix frame %3.3d", nSave-1);
      glutSetWindowTitle(winTitle);
    }
  }
  
  if(key == 'v'){
    if(vector_flag==1) vector_flag=0; else vector_flag=1;
    glutPostRedisplay();
  }

  if(key == 'q'){
    printf("Got q, so quitting\n");
    //glutDestroyWindow(win);
    exit(0);
  }
  if(key == 'g'){
    // Toggle grid display on/off
    if(display_grid==1) display_grid=0; else display_grid=1;
    glutPostRedisplay();
  }
  if(key == 'p'){
    plot_var++;
    if(plot_var > 3) plot_var=-1;
    if(plot_var==-1) printf("Plotting grid only\n");
    if(plot_var==0) printf("Plotting density\n");
    if(plot_var==1) printf("Plotting x velocity\n");
    if(plot_var==2) printf("Plotting y velocity\n");
    if(plot_var==3) printf("Plotting pressure\n");

    glutPostRedisplay();
  }
  if(key == 'h'){
    if(highlight_flag==1) highlight_flag=0; else highlight_flag=1;
    glutPostRedisplay();
  }
  if(key == 'b'){
    // Toggle boundary display
    if(boundary_flag==1) boundary_flag=0; else boundary_flag=1;
    glutPostRedisplay();
  }
  if(key == 's'){
    // Start/stop
    if(evolve_flag==1) evolve_flag=0; else evolve_flag=1;
    glutPostRedisplay();
  }
  if(key == 't'){
    // Toggle display triangle numbers
    if(display_triangle_numbers == 1) display_triangle_numbers = 0;
    else display_triangle_numbers = 1;
    glutPostRedisplay();
  }
  if(key == 'T'){
    vZoom = -1;

    int t = -1;
    printf("Enter triangle number: ");
    scanf("%i", &t);

    if (t >= 0 && t < n_triangle) {
      printf("Zooming in on triangle %i\n", t);
      tZoom = t;
    } else {
      tZoom = -1;
    }
  }
  if(key == 'V'){
    tZoom = -1;

    int v = -1;
    printf("Enter vertex number: ");
    scanf("%i", &v);

    if (v >= 0 && v < n_vertex) {
      printf("Zooming in on vertex %i\n", v);
      vZoom = v;
    }
  }
  
  if(key == 'c'){
    // Toggle copy
    if(copy_flag == 1) copy_flag = 0;
    else copy_flag = 1;
    glutPostRedisplay();
  }
  if(key == 'z'){
    // Zoom in on (xx, yy)
    float xx = (glut_current_maxx-glut_current_minx)*
      (float)x/(float)winSizeX + 
      glut_current_minx;
    float yy = (glut_current_miny-glut_current_maxy)*
      (float)y/(float)winSizeY + 
      glut_current_maxy;

    if (tZoom != -1) {
      float ax, bx, cx, ay, by, cy;
      GetTriangleCoordinates(tZoom, vertX, vertY,
			     &triVert[0*n_triangle],
			     &triVert[1*n_triangle],
			     &triVert[2*n_triangle],
			     n_vertex,
			     minx, maxx, miny, maxy,
			     &ax, &bx, &cx, &ay, &by, &cy);

      // Zoom in on (xx, yy)
      xx = (ax + bx + cx)/3.0;
      yy = (ay + by + cy)/3.0;
      printf("Zooming in on %f %f\n", xx, yy);
    }
    if (vZoom != -1) {
      // Zoom in on (xx, yy)
      xx = vertX[vZoom];
      yy = vertY[vZoom];
      printf("Zooming in on %f %f\n", xx, yy);
    }
    
    printf("Zooming in on %f %f\n", xx, yy);
    zoom_factor = 0.5;//*zoom_factor;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(xx-0.5*zoom_factor*(glut_current_maxx-glut_current_minx), 
	    xx+0.5*zoom_factor*(glut_current_maxx-glut_current_minx), 
	    yy-0.5*zoom_factor*(glut_current_maxy-glut_current_miny), 
	    yy+0.5*zoom_factor*(glut_current_maxy-glut_current_miny), 
	    -1.0, 1.0);  

    float gcminx = xx-0.5*zoom_factor*
      (glut_current_maxx-glut_current_minx);
    float gcmaxx = xx+0.5*zoom_factor*
      (glut_current_maxx-glut_current_minx);
    float gcminy = yy-0.5*zoom_factor*
      (glut_current_maxy-glut_current_miny);
    float gcmaxy = yy+0.5*zoom_factor*
      (glut_current_maxy-glut_current_miny);

    glut_current_minx = gcminx;
    glut_current_maxx = gcmaxx;
    glut_current_miny = gcminy;
    glut_current_maxy = gcmaxy;

    glutPostRedisplay();
  }
  if(key == 'Z'){
    // Zoom back out
    float xx = (glut_maxx-glut_minx)*(float)x/
      (float)winSizeX + glut_minx;
    float yy = (glut_miny-glut_maxy)*(float)y/
      (float)winSizeY + glut_maxy;

    //zoom_factor = zoom_factor*2.0;
    zoom_factor=1.0;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if(zoom_factor<1.0){
      glOrtho(xx-0.5*zoom_factor*(glut_maxx-glut_minx), 
	      xx+0.5*zoom_factor*(glut_maxx-glut_minx), 
	      yy-0.5*zoom_factor*(glut_maxy-glut_miny), 
	      yy+0.5*zoom_factor*(glut_maxy-glut_miny), 
	      -1.0, 1.0);  
    }else{
      glOrtho(glut_minx, glut_maxx, glut_miny, glut_maxy, -1.0, 1.0);  
      glut_current_minx = glut_minx;
      glut_current_maxx = glut_maxx;
      glut_current_miny = glut_miny;
      glut_current_maxy = glut_maxy;
    }
  }
}
