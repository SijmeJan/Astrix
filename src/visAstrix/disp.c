#include "visAstrix.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#define NEW_INDEX

#define SQ(X)		((X)*(X))		// Square macro
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


extern int nSave;
extern int win;

extern int winSizeX, winSizeY;

extern float glut_minx, glut_maxx, glut_miny, glut_maxy;
extern float glut_current_minx, glut_current_maxx, glut_current_miny, glut_current_maxy;

// Vertex coordinates
extern float *vertX, *vertY;
// Triangle and edge vertices
extern int *triVert, *triEdge;//, *edgeVert;
//extern int *edgeSegment;

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

extern int display_grid;
extern int display_triangle_numbers;
extern int evolve_flag;
extern int boundary_flag;
extern int highlight_flag;
extern int vector_flag;
extern float zoom_factor;
extern int plot_var;
extern int movie_flag;
extern int copy_flag;

float orient2d(float ax, float ay, float bx, float by, float cx, float cy)
{
  float detleft = (ax - cx) * (by - cy);
  float detright = (ay - cy) * (bx - cx);
  return detleft - detright;
}

float incircle(float ax, float ay,
	       float bx, float by,
	       float cx, float cy,
	       float dx, float dy)
{
  float adx, bdx, cdx, ady, bdy, cdy;
  float bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
  float alift, blift, clift;
  float det;

  adx = ax - dx;
  bdx = bx - dx;
  cdx = cx - dx;
  ady = ay - dy;
  bdy = by - dy;
  cdy = cy - dy;

  bdxcdy = bdx * cdy;
  cdxbdy = cdx * bdy;
  alift = adx * adx + ady * ady;

  cdxady = cdx * ady;
  adxcdy = adx * cdy;
  blift = bdx * bdx + bdy * bdy;

  adxbdy = adx * bdy;
  bdxady = bdx * ady;
  clift = cdx * cdx + cdy * cdy;

  det = alift * (bdxcdy - cdxbdy)
    + blift * (cdxady - adxcdy)
    + clift * (adxbdy - bdxady);
  
  return det;
}

float ColorTableRed(float c)
{
  if(c<0.33) return 0.0;
  if(c<0.67) return (c-0.33)/(0.67-0.33);
  return 1.0;
}
float ColorTableGreen(float c)
{
  if(c<0.33) return c/0.33;
  if(c<0.67) return 1.0;
  return 1.0-(c-0.67)/(1.0-0.67);
}
float ColorTableBlue(float c)
{
  if(c<0.33) return 1.0;
  if(c<0.67) return 1.0-(c-0.33)/(0.67-0.33);
  return 0.0;
}

int CanVertexBeTranslatedX(int a, int N)
{
  // Left, top:  -4*N
  // top:        -3*N
  // Right, top: -2*N
  // Left:       -1*N
  // Right:      +1*N
  // Left, bot:  +2*N
  // bot:        +3*N
  // Right, bot: +4*N

  // Can be translated to left?
  if (a >= 4*N ||
      (a >= N && a < 2*N) ||
      (a >= -2*N && a < -N)) return 1;

  // Can be translated to right?
  if (a < -3*N ||
      (a >= 2*N && a < 3*N) ||
      (a >=-N && a < 0)) return -1;

  // Neither left nor right
  return 0;
}
      
int CanVertexBeTranslatedY(int a, int N)
{
  // Can be translated to bottom?
  if (a < -N) return -1;

  // Can be translated to top?
  if (a >= 2*N) return 1;

  // Neither top nor bottom
  return 0;
}

void GetTriangleCoordinates(int i, float *pVertX, float *pVertY,
			    int *tv1, int *tv2, int *tv3, int nVertex,
			    float minx, float maxx, float miny, float maxy,
			    float *Ax, float *Bx, float *Cx,
			    float *Ay, float *By, float *Cy)
{
  int printFlag = 0;
  //if (i == 71) printFlag = 1;
  
  int a = tv1[i];
  int b = tv2[i];
  int c = tv3[i];

#ifdef NEW_INDEX

  float dxa = CanVertexBeTranslatedX(a, nVertex)*(maxx - minx);
  float dya = CanVertexBeTranslatedY(a, nVertex)*(maxy - miny);
  float dxb = CanVertexBeTranslatedX(b, nVertex)*(maxx - minx);
  float dyb = CanVertexBeTranslatedY(b, nVertex)*(maxy - miny);
  float dxc = CanVertexBeTranslatedX(c, nVertex)*(maxx - minx);
  float dyc = CanVertexBeTranslatedY(c, nVertex)*(maxy - miny);

  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  // Vertex coordinates    
  Ax[0] = pVertX[a] + dxa;
  Bx[0] = pVertX[b] + dxb;
  Cx[0] = pVertX[c] + dxc;
  Ay[0] = pVertY[a] + dya;
  By[0] = pVertY[b] + dyb;
  Cy[0] = pVertY[c] + dyc;

#else
  
  if (printFlag == 1)
    printf("Finding coordinates for triangle %i, vertices %i %i %i\n",
	   i, a, b, c);
  
  float dya = 0.0f;
  if (a >= 2*nVertex) {
    a -= 2*nVertex;
    dya = maxy - miny;
  }
  float dyb = 0.0f;
  if (b >= 2*nVertex) {
    b -= 2*nVertex;
    dyb = maxy - miny;
  }
  float dyc = 0.0f;
  if (c >= 2*nVertex) {
    c -= 2*nVertex;
    dyc = maxy - miny;
  }

  float dxa = 0.0f;
  if (a >= nVertex) {
    a -= nVertex;
    dxa = maxx - minx;
  }
  float dxb = 0.0f;
  if (b >= nVertex) {
    b -= nVertex;
    dxb = maxx - minx;
  }
  float dxc = 0.0f;
  if (c >= nVertex) {
    c -= nVertex;
    dxc = maxx - minx;
  }

  if (printFlag == 1)
    printf("dxy: %f %f %f %f %f %f\n",
	   dxa, dxb, dxc, dya, dyb, dyc);

  // Vertex coordinates    
  float ax, bx, cx, ay, by, cy;
  ax = pVertX[a];
  bx = pVertX[b];
  cx = pVertX[c];
  ay = pVertY[a];
  by = pVertY[b];
  cy = pVertY[c];

  float xa = ax;
  float xb = bx;
  float xc = cx;
  float ya = ay;
  float yb = by;
  float yc = cy;

  if (printFlag == 1)
    printf("First guess: %f %f %f %f %f %f\n", ax, bx, cx, ay, by, cy);
  
  if (xa == 0.5f*(maxx + minx)) xa = maxx + minx - 0.5f*(bx + cx);
  if (xb == 0.5f*(maxx + minx)) xb = maxx + minx - 0.5f*(ax + cx);
  if (xc == 0.5f*(maxx + minx)) xc = maxx + minx - 0.5f*(ax + bx);
  if (ya == 0.5f*(maxy + miny)) ya = maxy + miny - 0.5f*(by + cy);
  if (yb == 0.5f*(maxy + miny)) yb = maxy + miny - 0.5f*(ay + cy);
  if (yc == 0.5f*(maxy + miny)) yc = maxy + miny - 0.5f*(ay + by);
  
  ax -= sign(xa - 0.5f*(maxx + minx))*dxa;
  bx -= sign(xb - 0.5f*(maxx + minx))*dxb;
  cx -= sign(xc - 0.5f*(maxx + minx))*dxc;
  ay -= sign(ya - 0.5f*(maxy + miny))*dya;
  by -= sign(yb - 0.5f*(maxy + miny))*dyb;
  cy -= sign(yc - 0.5f*(maxy + miny))*dyc;

  if (printFlag == 1)
    printf("Translate: %f %f %f %f %f %f\n", ax, bx, cx, ay, by, cy);

  // Deal with triangle that wraps onto itself
  if (a == b || a == c || b == c) {
    if (dxa > 0.0) 
      if (orient2d(cx, cy, ax, ay, bx, by) < 0.0f)
	ax += 2.0f*sign(xa - 0.5f*(maxx + minx))*dxa;
    if (dxb > 0.0) 
      if (orient2d(cx, cy, ax, ay, bx, by) < 0.0f)
	bx += 2.0f*sign(xb - 0.5f*(maxx + minx))*dxb;
    if (dxc > 0.0) 
      if (orient2d(cx, cy, ax, ay, bx, by) < 0.0f)
	cx += 2.0f*sign(xc - 0.5f*(maxx + minx))*dxc;
    if (dya > 0.0) 
      if (orient2d(cx, cy, ax, ay, bx, by) < 0.0f)
	ay += 2.0f*sign(ya - 0.5f*(maxy + miny))*dya;
    if (dyb > 0.0) 
      if (orient2d(cx, cy, ax, ay, bx, by) < 0.0f)
	by += 2.0f*sign(yb - 0.5f*(maxy + miny))*dyb;
    if (dyc > 0.0) 
      if (orient2d(cx, cy, ax, ay, bx, by) < 0.0f)
	cy += 2.0f*sign(yc - 0.5f*(maxy + miny))*dyc;
  }

  if (printFlag == 1)
    printf("Wrap: %f %f %f %f %f %f\n", ax, bx, cx, ay, by, cy);

  int sa = sign(ax - 0.5f*(maxx + minx));
  int sb = sign(bx - 0.5f*(maxx + minx));
  int sc = sign(cx - 0.5f*(maxx + minx));
  int ta = sign(ay - 0.5f*(maxy + miny));
  int tb = sign(by - 0.5f*(maxy + miny));
  int tc = sign(cy - 0.5f*(maxy + miny));

  if (dxa > 0.0) {
    float axNew = ax + 2.0f*sign(xa - 0.5f*(maxx + minx))*dxa;
    if (sa != sb && sa != sc && sb == sc &&
	orient2d(cx, cy, axNew, ay, bx, by) > 0.0f &&
	orient2d(cx, cy, ax, ay, bx, by) < 0.0f) ax = axNew;
  }
  if (dxb > 0.0) {
    float bxNew = bx + 2.0f*sign(xb - 0.5f*(maxx + minx))*dxb;
    if (sb != sa && sb != sc && sa == sc &&
	orient2d(cx, cy, ax, ay, bxNew, by) > 0.0f &&
	orient2d(cx, cy, ax, ay, bx, by) < 0.0f) bx = bxNew;
  }
  if (dxc > 0.0) {
    float cxNew = cx + 2.0f*sign(xc - 0.5f*(maxx + minx))*dxc;
    if (sc != sa && sc != sb && sa == sb &&
	orient2d(cxNew, cy, ax, ay, bx, by) > 0.0f &&
	orient2d(cx, cy, ax, ay, bx, by) < 0.0f) cx = cxNew;
  }

  if (dya > 0.0) {
    float ayNew = ay + 2.0f*sign(ya - 0.5f*(maxy + miny))*dya;
    if (ta != tb && ta != tc && tb == tc &&
	orient2d(cx, cy, ax, ayNew, bx, by) > 0.0f &&
	orient2d(cx, cy, ax, ay, bx, by) < 0.0f) ay = ayNew;
  }
  if (dyb > 0.0) {
    float byNew = by + 2.0f*sign(yb - 0.5f*(maxy + miny))*dyb;
    if (tb != ta && tb != tc && ta == tc &&
	orient2d(cx, cy, ax, ay, bx, byNew) > 0.0f &&
	orient2d(cx, cy, ax, ay, bx, by) < 0.0f) by = byNew;
  }
  if (dyc > 0.0) {
    float cyNew = cy + 2.0f*sign(yc - 0.5f*(maxy + miny))*dyc;
    if (tc != ta && tc != tb && ta == tb &&
	orient2d(cx, cyNew, ax, ay, bx, by) > 0.0f &&
	orient2d(cx, cy, ax, ay, bx, by) < 0.0f) cy = cyNew;
  }
  
   if (printFlag == 1)
    printf("Illegal: %f %f %f %f %f %f\n", ax, bx, cx, ay, by, cy);
   
  Ax[0] = ax;
  Bx[0] = bx;
  Cx[0] = cx;
  Ay[0] = ay;
  By[0] = by;
  Cy[0] = cy;
#endif
  
} 

//###########################################################################
// 
//###########################################################################

void dispAstrix(void){
  int i, n;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  float *VertexColor = malloc(3*n_vertex*sizeof(float));
  for (i = 0; i < n_vertex; i++) {
    float s = 0.0;
    if(plot_var == 0)
      s = (vertDens[i] - mindens)/(maxdens - mindens + 1.0e-10);
    if(plot_var == 1)
      s = (vertVelx[i] - minvelx)/(maxvelx - minvelx + 1.0e-10);
    if(plot_var == 2)
      s = (vertVely[i] - minvely)/(maxvely - minvely + 1.0e-10);
    if(plot_var == 3)
      s = (vertPres[i] - minpres)/(maxpres - minpres + 1.0e-10);

    VertexColor[3*i+0] = ColorTableRed(s);
    VertexColor[3*i+1] = ColorTableGreen(s);
    VertexColor[3*i+2] = ColorTableBlue(s);

    if (plot_var == -1) {
      VertexColor[3*i+0] = 0.0;
      VertexColor[3*i+1] = 0.0;
      VertexColor[3*i+2] = 0.0;
    }
  }   
    
  float zValue = -1.0f;

  int maxCopy = 1;
  if (copy_flag == 1) maxCopy = 6;

  int nCopy;  
  for (nCopy = 0; nCopy < maxCopy; nCopy++) { 
    // Vertex and color arrays
    float *VertexArray = malloc(3*(ndim+1)*n_triangle*sizeof(float));
    float *ColorArray = malloc(3*(ndim+1)*n_triangle*sizeof(float));
    for (n = 0; n < n_triangle; n++) {
      float ax, bx, cx, ay, by, cy;
      GetTriangleCoordinates(n, vertX, vertY,
			     &triVert[0*n_triangle],
			     &triVert[1*n_triangle],
			     &triVert[2*n_triangle], n_vertex,
			     minx, maxx, miny, maxy,
			     &ax, &bx, &cx,
			     &ay, &by, &cy);

      if (nCopy == 1) {
	ax = ax + maxx - minx;
	bx = bx + maxx - minx;
	cx = cx + maxx - minx;
      }
      if (nCopy == 2) {
	ax = ax - maxx + minx;
	bx = bx - maxx + minx;
	cx = cx - maxx + minx;
      }
      if (nCopy == 3) {
	ay = ay + maxy - miny;
	by = by + maxy - miny;
	cy = cy + maxy - miny;
      }
      if (nCopy == 4) {
	ay = ay - maxy + miny;
	by = by - maxy + miny;
	cy = cy - maxy + miny;
      }
      if (nCopy == 5) {
	ax = ax + maxx - minx;
	bx = bx + maxx - minx;
	cx = cx + maxx - minx;
	ay = ay - maxy + miny;
	by = by - maxy + miny;
	cy = cy - maxy + miny;
      }
      
      int a = triVert[0*n_triangle + n];
      int b = triVert[1*n_triangle + n];
      int c = triVert[2*n_triangle + n];
      while (a >= n_vertex) a -= n_vertex;
      while (b >= n_vertex) b -= n_vertex;
      while (c >= n_vertex) c -= n_vertex;
      while (a < 0) a += n_vertex;
      while (b < 0) b += n_vertex;
      while (c < 0) c += n_vertex;
       
      VertexArray[3*(ndim+1)*n+0*(ndim+1)+0] = ax;
      VertexArray[3*(ndim+1)*n+0*(ndim+1)+1] = ay;
      VertexArray[3*(ndim+1)*n+0*(ndim+1)+2] = zValue;
      VertexArray[3*(ndim+1)*n+1*(ndim+1)+0] = bx;
      VertexArray[3*(ndim+1)*n+1*(ndim+1)+1] = by;
      VertexArray[3*(ndim+1)*n+1*(ndim+1)+2] = zValue;
      VertexArray[3*(ndim+1)*n+2*(ndim+1)+0] = cx;
      VertexArray[3*(ndim+1)*n+2*(ndim+1)+1] = cy;
      VertexArray[3*(ndim+1)*n+2*(ndim+1)+2] = zValue;
      
      ColorArray[3*(ndim+1)*n+0*(ndim+1)+0] = VertexColor[3*a+0];
      ColorArray[3*(ndim+1)*n+0*(ndim+1)+1] = VertexColor[3*a+1];
      ColorArray[3*(ndim+1)*n+0*(ndim+1)+2] = VertexColor[3*a+2];
      ColorArray[3*(ndim+1)*n+1*(ndim+1)+0] = VertexColor[3*b+0];
      ColorArray[3*(ndim+1)*n+1*(ndim+1)+1] = VertexColor[3*b+1];
      ColorArray[3*(ndim+1)*n+1*(ndim+1)+2] = VertexColor[3*b+2];
      ColorArray[3*(ndim+1)*n+2*(ndim+1)+0] = VertexColor[3*c+0];
      ColorArray[3*(ndim+1)*n+2*(ndim+1)+1] = VertexColor[3*c+1];
      ColorArray[3*(ndim+1)*n+2*(ndim+1)+2] = VertexColor[3*c+2];
      
      if (plot_var == -1) {
	float s = (triBlend[n] - minblend)/(maxblend - minblend);
	
	ColorArray[3*(ndim+1)*n+0*(ndim+1)+0] = ColorTableRed(s);
	ColorArray[3*(ndim+1)*n+0*(ndim+1)+1] = ColorTableGreen(s);
	ColorArray[3*(ndim+1)*n+0*(ndim+1)+2] = ColorTableBlue(s);
	ColorArray[3*(ndim+1)*n+1*(ndim+1)+0] = ColorTableRed(s);
	ColorArray[3*(ndim+1)*n+1*(ndim+1)+1] = ColorTableGreen(s);
	ColorArray[3*(ndim+1)*n+1*(ndim+1)+2] = ColorTableBlue(s);
	ColorArray[3*(ndim+1)*n+2*(ndim+1)+0] = ColorTableRed(s);
	ColorArray[3*(ndim+1)*n+2*(ndim+1)+1] = ColorTableGreen(s);
	ColorArray[3*(ndim+1)*n+2*(ndim+1)+2] = ColorTableBlue(s);
      }
    }
    
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    
    glColorPointer(3, GL_FLOAT, 0, ColorArray);
    glVertexPointer(3, GL_FLOAT, 0, VertexArray);
    glDrawArrays(GL_TRIANGLES, 0, 3*n_triangle);
    
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    
    free(VertexArray);
    free(ColorArray);
  }
  free(VertexColor);
  
  if (display_grid) {    
    int maxCopy = 1;
    if (copy_flag == 1) maxCopy = 6;

    int nCopy;  
    for (nCopy = maxCopy - 1; nCopy >= 0; nCopy--) { 
      glBegin(GL_LINES);
      for (n = 0; n < n_triangle; n++) {
	glColor3f(1.0f, 1.0f, 1.0f);
	if (nCopy > 0)
	  glColor3f(0.0f, 0.0f, 0.0f);

	float ax, bx, cx, ay, by, cy;
	GetTriangleCoordinates(n, vertX, vertY,
			       &triVert[0*n_triangle],
			       &triVert[1*n_triangle],
			       &triVert[2*n_triangle], n_vertex,
			       minx, maxx, miny, maxy,
			       &ax, &bx, &cx,
			       &ay, &by, &cy);
	
	if (nCopy == 1) {
	  ax = ax + maxx - minx;
	  bx = bx + maxx - minx;
	  cx = cx + maxx - minx;
	}
	if (nCopy == 2) {
	  ax = ax - maxx + minx;
	  bx = bx - maxx + minx;
	  cx = cx - maxx + minx;
	}
	if (nCopy == 3) {
	  ay = ay + maxy - miny;
	  by = by + maxy - miny;
	  cy = cy + maxy - miny;
	}
	if (nCopy == 4) {
	  ay = ay - maxy + miny;
	  by = by - maxy + miny;
	  cy = cy - maxy + miny;
	}
	if (nCopy == 5) {
	  ax = ax + maxx - minx;
	  bx = bx + maxx - minx;
	  cx = cx + maxx - minx;
	  ay = ay - maxy + miny;
	  by = by - maxy + miny;
	  cy = cy - maxy + miny;
	}
	
	glVertex3f(ax, ay, 0.0);
	glVertex3f(bx, by, 0.0);
	
	glVertex3f(bx, by, 0.0);
	glVertex3f(cx, cy, 0.0);
	
	glVertex3f(cx, cy, 0.0);
	glVertex3f(ax, ay, 0.0);
      }
      glEnd();  
    }
  }
  
  if (display_triangle_numbers) {
    for (n = 0; n < n_triangle; n++) {
      //int a = triVert[0*n_triangle + n];
      //int b = triVert[1*n_triangle + n];
      //int c = triVert[2*n_triangle + n];

      float ax, bx, cx, ay, by, cy;
      GetTriangleCoordinates(n, vertX, vertY,
			     &triVert[0*n_triangle],
			     &triVert[1*n_triangle],
			     &triVert[2*n_triangle], n_vertex,
			     minx, maxx, miny, maxy,
			     &ax, &bx, &cx,
			     &ay, &by, &cy);

      /*
      float ax, bx, cx, ay, by, cy;
      if (a < n_vertex) {
	ax = vertX[a];
	ay = vertY[a];
      }
      if (b < n_vertex) {
	bx = vertX[b];
	by = vertY[b];
      }
      if (c < n_vertex) {
	cx = vertX[c];
	cy = vertY[c];
      }
  
      if (a >= n_vertex) {
	a -= n_vertex;
	
	ax = vertX[a];
	ay = vertY[a];
	
	float Ap = orient2d(ax + maxx - minx, ay, bx, by, cx, cy);
	float Am = orient2d(ax - maxx + minx, ay, bx, by, cx, cy);
	
	if (Ap <= 0.0f) ax = ax - maxx + minx;
	if (Am <= 0.0f) ax = ax + maxx - minx;
	if (Ap > 0.0f && Am > 0.0f) {
	  if (Ap < Am) ax = ax + maxx - minx; else ax = ax - maxx + minx;
	}
      }
      if (b >= n_vertex) {
	b -= n_vertex;
	
	bx = vertX[b];
	by = vertY[b];
	
	float Ap = orient2d(ax, ay, bx + maxx - minx, by, cx, cy);
	float Am = orient2d(ax, ay, bx - maxx + minx, by, cx, cy);
	
	if (Ap <= 0.0f) bx = bx - maxx + minx;
	if (Am <= 0.0f) bx = bx + maxx - minx;
	if (Ap > 0.0f && Am > 0.0f) {
	  if (Ap < Am) bx = bx + maxx - minx; else bx = bx - maxx + minx;
	}
      }
      if (c >= n_vertex) {
	c -= n_vertex;
	
	cx = vertX[c];
	cy = vertY[c];
	
	float Ap = orient2d(ax, ay, bx, by, cx + maxx - minx, cy);
	float Am = orient2d(ax, ay, bx, by, cx - maxx + minx, cy);
	
	if (Ap <= 0.0f) cx = cx - maxx + minx;
	if (Am <= 0.0f) cx = cx + maxx - minx;
	if (Ap > 0.0f && Am > 0.0f) {
	  if (Ap < Am) cx = cx + maxx - minx; else cx = cx - maxx + minx;
	}
      }
      */
      
      glColor3f(1.0f, 1.0f, 1.0f);
 
      float dx = (ax + bx + cx)/3.0;
      float dy = (ay + by + cy)/3.0;

      // Triangle number
      glRasterPos2f(dx, dy);
      char tbuf1[30];
      sprintf(tbuf1, "%d", n);
      const char * p = tbuf1;
      do glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p); while(*(++p));

      glColor3f(1.0f, 0.0f, 0.0f);

      dx = (ax + bx)/2.0;
      dy = (ay + by)/2.0;

      // Edge number 1
      glRasterPos2f(dx, dy);
      char ebuf1[30];
      sprintf(ebuf1, "%d", triEdge[0*n_triangle + n]);
      p = ebuf1;
      do glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p); while(*(++p));

      dx = (bx + cx)/2.0;
      dy = (by + cy)/2.0;

      // Edge number 2
      glRasterPos2f(dx, dy);
      char ebuf2[30];
      sprintf(ebuf2, "%d", triEdge[1*n_triangle + n]);
      p = ebuf2;
      do glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p); while(*(++p));

      dx = (ax + cx)/2.0;
      dy = (ay + cy)/2.0;

      // Edge number 3
      glRasterPos2f(dx, dy);
      char ebuf3[30];
      sprintf(ebuf3, "%d", triEdge[2*n_triangle + n]);
      p = ebuf3;
      do glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p); while(*(++p));

      glColor3f(1.0f, 1.0f, 0.0f);

      dx = ax;
      dy = ay;

      // Vertex number 1
      glRasterPos2f(dx, dy);
      char vbuf1[30];
      sprintf(vbuf1, "%d", triVert[0*n_triangle + n]);
      p = vbuf1;
      do glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p); while(*(++p));

      dx = bx;
      dy = by;

      // Vertex number 2
      glRasterPos2f(dx, dy);
      char vbuf2[30];
      sprintf(vbuf2, "%d", triVert[1*n_triangle + n]);
      p = vbuf2;
      do glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p); while(*(++p));

      dx = cx;
      dy = cy;

      // Vertex number 3
      glRasterPos2f(dx, dy);
      char vbuf3[30];
      sprintf(vbuf3, "%d", triVert[2*n_triangle + n]);
      p = vbuf3;
      do glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p); while(*(++p));

    }
  }
  
  if(boundary_flag){
    glBegin(GL_LINES);
    glColor3f(1.0f,1.0f,1.0f);
    
    //float minx=0.0,maxx=1.0,miny=0.0,maxy=1.0;
    glVertex3f(minx,miny,0.0);
    glVertex3f(maxx,miny,0.0);
    
    glVertex3f(minx,miny,0.0);
    glVertex3f(minx,maxy,0.0);
      
    glVertex3f(maxx,maxy,0.0);
    glVertex3f(maxx,miny,0.0);
    
    glVertex3f(maxx,maxy,0.0);
    glVertex3f(minx,maxy,0.0);
    
    glEnd();
  }

  glutSwapBuffers();


  if(evolve_flag == 1){
    if(movie_flag == 1){
      printf("Taking screenshot %i\n", nSave-1);
      char fname[30];
      sprintf(fname, "movie_frames/shot%3.3d.tga", nSave-1);
      
      int nSize = winSizeX*winSizeY*3;
      
      GLubyte *pixels = malloc(nSize*sizeof(GLubyte));
      
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glReadPixels(0, 0, winSizeX, winSizeY, GL_BGR, GL_UNSIGNED_BYTE, pixels);
      
      unsigned char TGAheader[12] = {0,0,2,0,0,0,0,0,0,0,0,0};
      unsigned char header[6] = 
	{winSizeX%256, winSizeX/256, winSizeY%256, winSizeY/256, 24, 0};
      
      FILE *shot = fopen(fname, "wb");
      fwrite(TGAheader, sizeof(unsigned char), 12, shot);
      fwrite(header, sizeof(unsigned char), 6, shot);
      fwrite(pixels, sizeof(GLubyte), nSize, shot);
      fclose(shot);
      
      free(pixels);
      
    }

    char winTitle[25];
    sprintf(winTitle, "Astrix frame %3.3d", nSave-1);
    glutSetWindowTitle(winTitle);

    ReadFiles(0);
  }
}

void resizeAstrix(int w, int h)
{
  float xleft = 0.5*(glut_maxx + glut_minx) - 
    0.5*((float)w/(float)winSizeX)*(glut_maxx - glut_minx);
  float xright = 0.5*(glut_maxx + glut_minx) +
    0.5*((float)w/(float)winSizeX)*(glut_maxx - glut_minx);

  float yleft = 0.5*(glut_maxy + glut_miny) - 
    0.5*((float)h/(float)winSizeY)*(glut_maxy - glut_miny);
  float yright = 0.5*(glut_maxy + glut_miny) + 
    0.5*((float)h/(float)winSizeY)*(glut_maxy - glut_miny);
 
  glut_minx = xleft;
  glut_maxx = xright;
  glut_miny = yleft;
  glut_maxy = yright;

  xleft = 0.5*(glut_current_maxx + glut_current_minx) - 
    0.5*((float)w/(float)winSizeX)*(glut_current_maxx - glut_current_minx);
  xright = 0.5*(glut_current_maxx + glut_current_minx) +
    0.5*((float)w/(float)winSizeX)*(glut_current_maxx - glut_current_minx);

  yleft = 0.5*(glut_current_maxy + glut_current_miny) - 
    0.5*((float)h/(float)winSizeY)*(glut_current_maxy - glut_current_miny);
  yright = 0.5*(glut_current_maxy + glut_current_miny) + 
    0.5*((float)h/(float)winSizeY)*(glut_current_maxy - glut_current_miny);

  glut_current_minx = xleft;
  glut_current_maxx = xright;
  glut_current_miny = yleft;
  glut_current_maxy = yright;

  winSizeX = w;
  winSizeY = h;
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(glut_current_minx, glut_current_maxx,
	  glut_current_miny, glut_current_maxy, -1.0, 1.0);  

  glViewport(0, 0, w, h);
  printf("%i %i %f %f \n", w, h, glut_minx, glut_maxx);
}
