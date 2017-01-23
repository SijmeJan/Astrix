/*! \file triangleLow.h
\brief Header file for a few low-level triangle functions

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef ASTRIX_TRIANGLE_LOW_H
#define ASTRIX_TRIANGLE_LOW_H

namespace astrix {

//##############################################################################
/*! Assume that \a b and \a f are periodic variants of each other, so that they are a distance of \a dx and/or \a dy apart. Now move vertex (\a x, \a y) over that same distance so that it is in the same part of the domain as \a b.

\param b Vertex index to move to
\param f Vertex index to move from
\param dx Periodic domain size x
\param dy Periodic domain size y
\param N Total number of vertices
\param x x coordinate of vertex to be moved (output)
\param y y coordinate of vertex to be moved (output)*/
//##############################################################################

__host__ __device__ inline
void TranslateVertexToVertex(const int b, const int f,
                             const real dx, const real dy, const int N,
                             real& x, real& y)
{
  if (b - f == 4*N || b - f == 1*N || b - f == -2*N) x += dx;
  if (b - f == 2*N || b - f == -1*N || b - f == -4*N) x -= dx;
  if (b - f == 4*N || b - f == 3*N || b - f == 2*N) y += dy;
  if (b - f == -2*N || b - f == -3*N || b - f == -4*N) y -= dy;
}

//##############################################################################
/*! Assume that \a v or a periodic version of \a v belongs to triangle. Now move the triangle so that it becomes local to \a v.

\param v Vertex index to move to
\param dx Periodic domain size x
\param dy Periodic domain size y
\param N Total number of vertices
\param a First vertex of triangle
\param b Second vertex of triangle
\param c Third vertex of triangle
\param ax First vertex x coordinate (output)
\param ay First vertex y coordinate (output)
\param bx Second vertex x coordinate (output)
\param by Second vertex y coordinate (output)
\param cx Third vertex x coordinate (output)
\param cy Third vertex y coordinate (output)*/
//##############################################################################

__host__ __device__ inline
void TranslateTriangleToVertex(const int v, const real dx, const real dy,
                               const int N,
                               int a, const int b, const int c,
                               real& ax, real& ay, real& bx, real& by,
                               real& cx, real& cy)
{
  if (a != v && b != v && c != v) {
    if (a == v + N || b == v + N || c == v + N ||
        a == v + 4*N || b == v + 4*N || c == v + 4*N ||
        a == v - 2*N || b == v - 2*N || c == v - 2*N) {
      ax -= dx;
      bx -= dx;
      cx -= dx;
    }
    if (a == v - N || b == v - N || c == v - N ||
        a == v - 4*N || b == v - 4*N || c == v - 4*N ||
        a == v + 2*N || b == v + 2*N || c == v + 2*N) {
      ax += dx;
      bx += dx;
      cx += dx;
    }
    if (a == v + 3*N || b == v + 3*N || c == v + 3*N ||
        a == v + 4*N || b == v + 4*N || c == v + 4*N ||
        a == v + 2*N || b == v + 2*N || c == v + 2*N) {
      ay -= dy;
      by -= dy;
      cy -= dy;
    }
    if (a == v - 3*N || b == v - 3*N || c == v - 3*N ||
        a == v - 4*N || b == v - 4*N || c == v - 4*N ||
        a == v - 2*N || b == v - 2*N || c == v - 2*N) {
      ay += dy;
      by += dy;
      cy += dy;
    }
  }
}

//##############################################################################
/*! \brief Check if vertex can be translated in x-direction

  Check if vertex \a a can be translated (i.e. its index changed) in the x direction. Returns 1 if it can in the positive x direction, -1 if it can in the negative x direction, zero otherwise.

\param a Vertex index
\param N Total number of vertices*/
//##############################################################################

__host__ __device__ inline
real CanVertexBeTranslatedX(const int a, const int N)
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
      (a >= -2*N && a < -N)) return (real)1.0;

  // Can be translated to right?
  if (a < -3*N ||
      (a >= 2*N && a < 3*N) ||
      (a >=-N && a < 0)) return (real) -1.0;

  // Neither left nor right
  return (real) 0.0;
}

//##############################################################################
/*! \brief Check if vertex can be translated in y-direction

  Check if vertex \a a can be translated (i.e. its index changed) in the y direction. Returns 1 if it can in the positive y direction, -1 if it can in the negative y direction, zero otherwise.

\param a Vertex index
\param N Total number of vertices*/
//##############################################################################

__host__ __device__ inline
real CanVertexBeTranslatedY(const int a, const int N)
{
  // Can be translated to bottom?
  if (a < -N) return (real) -1.0;

  // Can be translated to top?
  if (a >= 2*N) return (real) 1.0;

  // Neither top nor bottom
  return (real) 0.0;
}

//######################################################################
/*! If a triangle is invalid, rotate indices such that it becomes valid.

\param a First vertex (may be modified)
\param b Second vertex (may be modified)
\param c Third vertex (may be modified)
\param N Total number of vertices in Mesh*/
//######################################################################

__host__ __device__ inline
void MakeValidIndices(int& a, int& b, int& c, const int N)
{
  real f =
    CanVertexBeTranslatedX(a, N) +
    CanVertexBeTranslatedX(b, N) +
    CanVertexBeTranslatedX(c, N);

  if (f < (real)-1.0) {
    a += N;
    b += N;
    c += N;
  }

  if (f > (real) 1.0) {
    a -= N;
    b -= N;
    c -= N;
  }

  f =
    CanVertexBeTranslatedY(a, N) +
    CanVertexBeTranslatedY(b, N) +
    CanVertexBeTranslatedY(c, N);

  if (f < (real)-1.0) {
    a += 3*N;
    b += 3*N;
    c += 3*N;
  }

  if (f > (real) 1.0) {
    a -= 3*N;
    b -= 3*N;
    c -= 3*N;
  }
}

//######################################################################
/*! Find coordinates of triangle. This is non-trivial in the case of a periodic domain.

\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param a Triangle's first vertex
\param b Triangle's second vertex
\param c Triangle's third vertex
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param ax Place to store first x coordinate
\param bx Place to store second x coordinate
\param cx Place to store third x coordinate
\param ay Place to store first y coordinate
\param by Place to store second y coordinate
\param cy Place to store third y coordinate */
//######################################################################

__host__ __device__ inline
void GetTriangleCoordinates(const real2* __restrict__ pVc,
                            int a, int b, int c,
                            const int nVertex, const real Px, const real Py,
                            real& ax, real& bx, real& cx,
                            real& ay, real& by, real& cy)
{
  real dxa = CanVertexBeTranslatedX(a, nVertex)*Px;
  real dya = CanVertexBeTranslatedY(a, nVertex)*Py;
  real dxb = CanVertexBeTranslatedX(b, nVertex)*Px;
  real dyb = CanVertexBeTranslatedY(b, nVertex)*Py;
  real dxc = CanVertexBeTranslatedX(c, nVertex)*Px;
  real dyc = CanVertexBeTranslatedY(c, nVertex)*Py;

  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  // Vertex coordinates
  ax = pVc[a].x + dxa;
  bx = pVc[b].x + dxb;
  cx = pVc[c].x + dxc;
  ay = pVc[a].y + dya;
  by = pVc[b].y + dyb;
  cy = pVc[c].y + dyc;
}

//######################################################################
/*! Find coordinates of a single vertex of a triangle. This is non-trivial in the case of a periodic domain.

\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param a Triangle vertex
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param ax Place to store first x coordinate
\param ay Place to store first y coordinate*/
//######################################################################

__host__ __device__ inline
void GetTriangleCoordinatesSingle(const real2* __restrict__ pVc,
                                  int a, const int nVertex,
                                  const real Px, const real Py,
                                  real& ax, real& ay)
{
  real dxa = CanVertexBeTranslatedX(a, nVertex)*Px;
  real dya = CanVertexBeTranslatedY(a, nVertex)*Py;

  while (a >= nVertex) a -= nVertex;
  while (a < 0) a += nVertex;

  // Vertex coordinates
  ax = pVc[a].x + dxa;
  ay = pVc[a].y + dya;
}

}  // namespace astrix

#endif  // ASTRIX_TRIANGLE_LOW_H
