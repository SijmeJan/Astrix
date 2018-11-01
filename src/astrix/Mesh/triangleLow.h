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
/*! Normalize vertex index \a so that the 0 < return value < \a N

\param a Vertex index to normalize
\param N Total number of vertices in Mesh*/
//##############################################################################

__host__ __device__ inline
int NormalizeVertex(int a, const int N)
{
  while (a >= N) a -= N;
  while (a < 0) a += N;

  return a;
}

//##############################################################################
/*! Normalize vertex indices \a V so that the 0 < return value < \a N

\param V Vertex indices to normalize
\param N Total number of vertices in Mesh*/
//##############################################################################

__host__ __device__ inline
int3 NormalizeVertex(int3 V, const int N)
{
  while (V.x >= N) V.x -= N;
  while (V.x < 0) V.x += N;
  while (V.y >= N) V.y -= N;
  while (V.y < 0) V.y += N;
  while (V.z >= N) V.z -= N;
  while (V.z < 0) V.z += N;

  return V;
}

//##############################################################################
/*! Return triangle vertex not part of edge \a e. Returns -1 if \a e not part of triangle.

\param e Edge to consider
\param V Vertices of triangle
\param E Edges of triangle*/
//##############################################################################

__host__ __device__ inline
int VertexNotPartOfEdge(const int e, const int3 V, const int3 E)
{
  int ret = -1;
  if (e == E.x) ret = V.z;
  if (e == E.y) ret = V.x;
  if (e == E.z) ret = V.y;

  return ret;
}

//##############################################################################
/*! Given the edge lengths of a triangle, compute its area using Heron's formula

\param l1 Edge length 1
\param l2 Edge length 2
\param l3 Edge length 3*/
//##############################################################################

__host__ __device__ inline
real TriangleAreaFromSides(real l1, real l2, real l3)
{
  // semiperimeter
  real s = (real) 0.5*(l1 + l2 + l3);

  // Return area through Heron's formula
  return sqrt(s*(s - l1)*(s - l2)*(s - l3));
}

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
  if (b - f == 4*N  || b - f == 1*N  || b - f == -2*N) x += dx;
  if (b - f == 2*N  || b - f == -1*N || b - f == -4*N) x -= dx;
  if (b - f == 4*N  || b - f == 3*N  || b - f == 2*N)  y += dy;
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

\param *pVc Pointer to vertex coordinates
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

\param *pVc Pointer to vertex coordinates
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

//######################################################################
/*! Find vertices belonging to edge \a e.

Find vertices belonging to edge \a e, knowing it is part of triangle \a t with edges \a E and vertice \a V. Find the vertices \a a and \a b belonging to edge \a e in counterclockwise order as seen from \a t. Both \a a and \a b are output.

\param e Edge to find vertices for
\param t Triangle that \a e is part of
\param V Vertices of \a t
\param E Edges of \a t
\param a Place to store first vertex of \a e
\param b Place to store second vertex of \a e*/
//######################################################################

__host__ __device__ inline
void GetEdgeVertices(const int e, const int t, const int3 V,
                     const int3 E, int& a, int& b)
{
  if (e == E.x) {
    a = V.x;
    b = V.y;
  }
  if (e == E.y) {
    a = V.y;
    b = V.z;
  }
  if (e == E.z) {
    a = V.z;
    b = V.x;
  }
}

//######################################################################
/*! Find vertices belonging to edge \a e.

Find vertices belonging to edge \a e, knowing it is part of triangle \a t with edges \a E. Find the vertices \a a and \a b belonging to edge \a e in counterclockwise order as seen from \a t. Both \a a and \a b are output.

\param e Edge to find vertices for
\param *pTv Pointer to triangle vertices
\param t Triangle that \a e is part of
\param V Place to store output vertices
\param E Edges of \a t
\param a Place to store first vertex of \a e
\param b Place to store second vertex of \a e*/
//######################################################################

__host__ __device__ inline
void GetEdgeVertices(const int e,
                     const int3* __restrict__ pTv,
                     const int t, int3& V, const int3 E, int& a, int& b)
{
  V = pTv[t];
  GetEdgeVertices(e, t, V, E, a, b);
}

//######################################################################
/*! Find vertices belonging to edge \a e.

Find vertices belonging to edge \a e, knowing it is part of triangle \a t. First, determine the edges of this triangle \a E, then find the vertices \a a and \a b belonging to edge \a e in counterclockwise order as seen from \a t. Both \a E and \a a and \a b are output.

\param e Edge to find vertices for
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param t Triangle that \a e is part of
\param V Place to store output vertices
\param E Place to store output edges
\param a Place to store first vertex of \a e
\param b Place to store second vertex of \a e*/
//######################################################################

__host__ __device__ inline
void GetEdgeVertices(const int e,
                     const int3* __restrict__ pTv,
                     const int3* __restrict__ pTe,
                     const int t, int3& V, int3& E, int& a, int& b)
{
  // Edges of triangle
  E = pTe[t];

  GetEdgeVertices(e, pTv, t, V, E, a, b);
}

//######################################################################
/*! Find vertices belonging to edge \a e.

Find vertices belonging to edge \a e. First, find a triangle \a t neighbouring \a e, then determine the edges of this triangle \a E, and finally find the vertices \a a and \a b belonging to edge \a e in counterclockwise order as seen from \a t. Both \a t, \a E and \a a and \a b are output.

\param e Edge to find vertices for
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param t Place to store output triangle
\param E Place to store outpud edges
\param a Place to store first vertex of \a e
\param b Place to store second vertex of \a e*/
//######################################################################

__host__ __device__ inline
void GetEdgeVertices(const int e,
                     const int3* __restrict__ pTv,
                     const int3* __restrict__ pTe,
                     const int2* __restrict__ pEt,
                     int2& tCollapse, int3& V, int3& E, int& a, int& b)
{
  // Find neighbouring triangle
  tCollapse = pEt[e];
  if (tCollapse.x == -1) {
    tCollapse.x = tCollapse.y;
    tCollapse.y = -1;
  }

  GetEdgeVertices(e, pTv, pTe, tCollapse.x, V, E, a, b);
}

//######################################################################
/*! Return triangle in \a T that is not \a t. Usually \a T is a pair of triangles next to an edge.

\param t Triangle we do not want
\param T Pair of triangles, one of which is \a t*/
//######################################################################

__host__ __device__ inline
int OtherNeighbouringTriangle(const int t, const int2 T)
{
  int ret = T.x;
  if (ret == t) ret = T.y;
  return ret;
}

//######################################################################
/*! Return edge of triangle in counterclockwise direction from edge \a e.

\param e Edge to consider
\param E Edges of triangle*/
//######################################################################

__host__ __device__ inline
int NextEdgeCounterClockwise(const int e, const int3 E)
{
  if (e == E.x) return E.y;
  if (e == E.y) return E.z;
  if (e == E.z) return E.x;
  return -1;
}

//######################################################################
/*! Return edge of triangle in clockwise direction from edge \a e.

\param e Edge to consider
\param E Edges of triangle*/
//######################################################################

__host__ __device__ inline
int NextEdgeClockwise(const int e, const int3 E)
{
  if (e == E.x) return E.z;
  if (e == E.y) return E.x;
  if (e == E.z) return E.y;
  return -1;
}

//#########################################################################
/*! Move counterclockwise from edge \a eStart until we hit a segment, the index of which is returned. If no segment is found, return -1.

\param eStart Edge to start from
\param t Triangle to start from (will be updated!)
\param E Edges for triangle \a t (will be updated!)
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles*/
//#########################################################################

__host__ __device__ inline
int FindSegmentCounterClock(int eStart, int& t, int3& E,
                            int3 *pTe, int2 *pEt)
{
  int eCross = eStart;   // Starting on this edge
  int tStart = t;        // Starting in this triangle

  while (1) {
    eCross = NextEdgeClockwise(eCross, E);

    // Triangle to move into
    t = OtherNeighbouringTriangle(t, pEt[eCross]);

    // Done if eCross is a segment or back at beginning
    if (t == -1 || t == tStart) {
      // If no segment found, return -1
      if (t == tStart) eCross = -1;
      if (t == -1)
        t = OtherNeighbouringTriangle(t, pEt[eCross]);
      break;
    }

    // Edges of current triangle
    E = pTe[t];
  }

  // Return segment
  return eCross;
}

//#########################################################################
/*! Move counterclockwise from edge \a eStart until we hit a segment, the index of which is returned. If no segment is found, return -1.

\param eStart Edge to start from
\param t Triangle to start from (will be updated!)
\param E Edges for triangle \a t (will be updated!)
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles*/
//#########################################################################

__host__ __device__ inline
int FindSegmentClock(int eStart, int& t, int3& E,
                     int3 *pTe, int2 *pEt)
{
  int eCross = eStart;   // Starting on this edge
  int tStart = t;        // Starting in this triangle

  while (1) {
    eCross = NextEdgeCounterClockwise(eCross, E);

    // Triangle to move into
    t = OtherNeighbouringTriangle(t, pEt[eCross]);

    // Done if eCross is a segment or back at beginning
    if (t == -1 || t == tStart) {
      // If no segment found, return -1
      if (t == tStart) eCross = -1;
      if (t == -1)
        t = OtherNeighbouringTriangle(t, pEt[eCross]);
      break;
    }

    // Edges of current triangle
    E = pTe[t];
  }

  // Return segment
  return eCross;
}

//#########################################################################
/*! Walk around, visiting all triangles surrounding edge \a eTest. Triangle \a t and edge crossed \a eCross will be updated.

\param eTest Edge to walk around
\param isSegment Flag whether \a eTest is a segment
\param t Current triangle, will be updated
\param eCross Edge crossed, will be updated
\param E Edges of current triangle
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles*/
//#########################################################################

__host__ __device__ inline
void WalkAroundEdge(const int eTest,
                    const int isSegment,
                    int& t, int& eCross, int3 E,
                    int3 *pTe, int2 *pEt)
{
  // Edge to cross into next triangle
  eCross = NextEdgeCounterClockwise(eCross, E);
  if (eCross == eTest)
    eCross = NextEdgeCounterClockwise(eCross, E);

  // Move into next triangle
  t = OtherNeighbouringTriangle(t, pEt[eCross]);

  if (t == -1) {
    t = OtherNeighbouringTriangle(t, pEt[eCross]);

    // Go counterclockwise to find another segment to continue
    eCross = FindSegmentCounterClock(eCross, t, E, pTe, pEt);

    // If moving along segment, repeat so that we can continue clockwise
    if (isSegment)
      eCross = FindSegmentCounterClock(eCross, t, E, pTe, pEt);
  }
}


}  // namespace astrix

#endif  // ASTRIX_TRIANGLE_LOW_H
