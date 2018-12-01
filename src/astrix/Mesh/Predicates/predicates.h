/*! \file predicates.h
\brief Header file containing Predicates class definition

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef ASTRIX_PREDICATES_H
#define ASTRIX_PREDICATES_H

namespace astrix {

// Forward declaration Array
template <class T> class Array;
class Device;

//! Class for exact geometric predicates
/*! Creating and updating Delaunay triangulations requires exact evaluation of certain geometric tests, in particular whether a point d lies inside or outside the circle through three other points a, b and c, and whether three points a, b and c are orientated in anti-clockwise direction or not. This class uses the algorithms by Shewchuck (1997).*/
class Predicates
{
 public:
  //! Constructor
  /*! Construct Predicates object and compute parameter vector on host and device.
    \param *device Pointer to Device object.*/
  explicit Predicates(Device *device);
  //! Destructor
  ~Predicates();

  //! Test whether point d lies in circle formed by a, b and c.
  /*! Test whether point (\a dx, \a dy) lies inside a circle defined by points (\a ax, \a ay), (\a bx, \a by) and (\a cx, \a cy). Return value > 0 if d lies inside circle, < 0 if outside circle, and = 0 if exactly on circle.
    \param ax x coordinate of point a
    \param ay y coordinate of point a
    \param bx x coordinate of point b
    \param by y coordinate of point b
    \param cx x coordinate of point c
    \param cy y coordinate of point c
    \param dx x coordinate of point d
    \param dy y coordinate of point d
    \param *pParam pointer to parameter vector
  */
  __host__ __device__
  real incircle(real ax, real ay,
                real bx, real by,
                real cx, real cy,
                real dx, real dy,
                const real * const pParam) const;

  //! Test whether points a, b and c lie in counterclockwise orientation.
  /*! Test whether points (\a ax, \a ay), (\a bx, \a by) and (\a cx, \a cy) are orientated in counterclockwise direction. Return value > 0 if a, b, c occur in counterclockwise order, < 0 if in clockwise order, and = 0 if points are collinear.
    \param ax x coordinate of point a
    \param ay y coordinate of point a
    \param bx x coordinate of point b
    \param by y coordinate of point b
    \param cx x coordinate of point c
    \param cy y coordinate of point c
    \param *pParam pointer to parameter vector
  */
  __host__ __device__
  real orient2d(real ax, real ay,
                real bx, real by,
                real cx, real cy,
                const real * const pParam) const;

  //! Get pointer to parameter vector
  /*! Return pointer to parameter vector, either device pointer (if \a cudaFlag = 1) or host pointer (if cudaFlag = 0). Note that both exist unless there is no CUDA capable device.
    \param cudaFlag Flag whether to return device pointer (= 1) or host pointer (= 0).
  */
  real* GetParamPointer(int cudaFlag) const;

 private:
  //! Parameter vector.
  Array<real> *param;

  //! Return absolute value of \a a
  /*! Compute absolute value of \a a
    \param a Value we need the absolute value of
  */
  __host__ __device__
  inline real Absolute(real a) const
  {
    return ((a) >= (real) 0.0 ? (a) : -(a));
  }

  __host__ __device__
  real incircleadapt(real ax, real ay,
                     real bx, real by,
                     real cx, real cy,
                     real dx, real dy,
                     real permanent,
                     const real * const pParam) const;
  __host__ __device__
  real orient2dadapt(real ax, real ay,
                     real bx, real by,
                     real cx, real cy,
                     real detsum,
                     const real * const pParam) const;

  /***************************************************************************/
  /*                                                                         */
  /*  fast_expansion_sum_zeroelim()   Sum two expansions, eliminating zero   */
  /*                                  components from the output expansion.  */
  /*                                                                         */
  /*  Sets h = e + f.  See the long version of my paper for details.         */
  /*                                                                         */
  /*  If round-to-even is used (as with IEEE 754), maintains the strongly    */
  /*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h */
  /*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent    */
  /*  properties.                                                            */
  /*                                                                         */
  /***************************************************************************/
  __host__ __device__
    int fast_expansion_sum_zeroelim(int elen, real *e, int flen,
                                    real *f, real *h) const
  // h cannot be e or f.
  {
    real zero = 0.0;
    real Q;
    real Qnew;
    real hh;
    int eindex, findex, hindex;
    real enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
      Q = enow;
      enow = e[++eindex];
    } else {
      Q = fnow;
      fnow = f[++findex];
    }
    hindex = 0;
    if ((eindex < elen) && (findex < flen)) {
      if ((fnow > enow) == (fnow > -enow)) {
        Fast_Two_Sum(enow, Q, Qnew, hh);
        enow = e[++eindex];
      } else {
        Fast_Two_Sum(fnow, Q, Qnew, hh);
        fnow = f[++findex];
      }
      Q = Qnew;
      if (hh != zero) {
        h[hindex++] = hh;
      }
      while ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
          Two_Sum(Q, enow, Qnew, hh);
          enow = e[++eindex];
        } else {
          Two_Sum(Q, fnow, Qnew, hh);
          fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != zero) {
          h[hindex++] = hh;
        }
      }
    }
    while (eindex < elen) {
      Two_Sum(Q, enow, Qnew, hh);
      enow = e[++eindex];
      Q = Qnew;
      if (hh != zero) {
        h[hindex++] = hh;
      }
    }
    while (findex < flen) {
      Two_Sum(Q, fnow, Qnew, hh);
      fnow = f[++findex];
      Q = Qnew;
      if (hh != zero) {
        h[hindex++] = hh;
      }
    }
    if ((Q != zero) || (hindex == 0)) {
      h[hindex++] = Q;
    }
    return hindex;
  }

  /***************************************************************************/
  /*                                                                         */
  /*  scale_expansion_zeroelim()   Multiply an expansion by a scalar,        */
  /*                               eliminating zero components from the      */
  /*                               output expansion.                         */
  /*                                                                         */
  /*  Sets h = be.  See either version of my paper for details.              */
  /*                                                                         */
  /*  Maintains the nonoverlapping property.  If round-to-even is used (as   */
  /*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent  */
  /*  properties as well.  (That is, if e has one of these properties, so    */
  /*  will h.)                                                               */
  /*                                                                         */
  /***************************************************************************/

  __host__ __device__
    inline int scale_expansion_zeroelim(int elen, real *e,
                                        real b, real *h,
                                        const real * const pParam) const
  /* e and h cannot be the same. */
  {
    real zero = 0.0;

    real Q, sum;
    real hh;
    real product1;
    real product0;
    int eindex, hindex;
    real enow;
    real ahi, alo, bhi, blo;

    Split(b, bhi, blo, pParam[0]);
    Two_Product_Presplit(e[0], b, bhi, blo, Q, hh, pParam[0], ahi, alo);
    hindex = 0;
    if (hh != 0) {
      h[hindex++] = hh;
    }
    for (eindex = 1; eindex < elen; eindex++) {
      enow = e[eindex];
      Two_Product_Presplit(enow, b, bhi, blo, product1, product0,
                           pParam[0], ahi, alo);
      Two_Sum(Q, product0, sum, hh);
      if (hh != 0) {
        h[hindex++] = hh;
      }
      Fast_Two_Sum(product1, sum, Q, hh);
      if (hh != 0) {
        h[hindex++] = hh;
      }
    }
    if ((Q != zero) || (hindex == 0)) {
      h[hindex++] = Q;
    }
    return hindex;
  }

  /***************************************************************************/
  /*                                                                         */
  /*  estimate()   Produce a one-word estimate of an expansion's value.      */
  /*                                                                         */
  /*  See either version of my paper for details.                            */
  /*                                                                         */
  /***************************************************************************/

  __host__ __device__
    inline real estimate(int elen, real *e) const
  {
    real Q;
    int eindex;

    Q = e[0];
    for (eindex = 1; eindex < elen; eindex++) {
      Q += e[eindex];
    }
    return Q;
  }

  // Many of the operations are broken up into two pieces, a main part that
  // performs an approximate operation, and a "tail" that computes the
  // roundoff error of that operation.

  __host__ __device__
    inline void Fast_Two_Sum_Tail(real a, real b, real x, real& y) const
  {
    real bvirt = x - a;
    y = b - bvirt;
  }

  __host__ __device__
    inline void Fast_Two_Sum(real a, real b, real& x, real& y) const
  {
    x = (real) (a + b);
    Fast_Two_Sum_Tail(a, b, x, y);
  }

  __host__ __device__
    inline void Fast_Two_Diff_Tail(real a, real b, real x, real& y) const
  {
    real bvirt = a - x;
    y = bvirt - b;
  }

  __host__ __device__
    inline void Fast_Two_Diff(real a, real b, real& x, real& y) const
  {
    x = (real) (a - b);
    Fast_Two_Diff_Tail(a, b, x, y);
  }

  __host__ __device__
    inline void Two_Sum_Tail(real a, real b, real x, real& y) const
  {
    real bvirt = (real) (x - a);
    real avirt = x - bvirt;
    real bround = b - bvirt;
    real around = a - avirt;
    y = around + bround;
  }

  __host__ __device__
    inline void Two_Sum(real a, real b, real& x, real& y) const
  {
    x = (real) (a + b);
    Two_Sum_Tail(a, b, x, y);
  }

  __host__ __device__
    inline void Two_Diff_Tail(real a, real b, real x, real& y) const
  {
    real bvirt = (real) (a - x);
    real avirt = x + bvirt;
    real bround = bvirt - b;
    real around = a - avirt;
    y = around + bround;
  }

  __host__ __device__
    inline void Two_Diff(real a, real b, real& x, real& y) const
  {
    x = (real) (a - b);
    Two_Diff_Tail(a, b, x, y);
  }

  __host__ __device__
    inline void Split(real a, real& ahi, real& alo, real splitter) const
  {
    real c = (real) (splitter * a);
    real abig = (real) (c - a);
    ahi = c - abig;
    alo = a - ahi;
  }

  __host__ __device__
    inline void Two_Product_Tail(real a, real b,
                                 real x, real& y,
                                 real splitter,
                                 real& ahi, real& alo,
                                 real& bhi, real& blo) const
  {
    Split(a, ahi, alo, splitter);
    Split(b, bhi, blo, splitter);
    real err1 = x - (ahi * bhi);
    real err2 = err1 - (alo * bhi);
    real err3 = err2 - (ahi * blo);
    y = (alo * blo) - err3;
  }

  __host__ __device__
    inline void Two_Product(real a, real b,
                            real& x, real& y,
                            real splitter,
                            real& ahi, real& alo,
                            real& bhi, real& blo) const
  {
    x = (real) (a * b);
    Two_Product_Tail(a, b, x, y, splitter, ahi, alo, bhi, blo);
  }

  __host__ __device__
    inline void Two_Product_Presplit(real a, real b,
                                     real bhi, real blo,
                                     real& x, real& y, real splitter,
                                     real& ahi, real& alo) const
  {
    x = (real) (a * b);
    Split(a, ahi, alo, splitter);
    real err1 = x - (ahi * bhi);
    real err2 = err1 - (alo * bhi);
    real err3 = err2 - (ahi * blo);
    y = (alo * blo) - err3;
  }

  __host__ __device__
    inline void Two_Product_2Presplit(real a, real ahi, real alo,
                                      real b, real bhi, real blo,
                                      real& x, real& y) const
  {
    x = (real) (a * b);
    real err1 = x - (ahi * bhi);
    real err2 = err1 - (alo * bhi);
    real err3 = err2 - (ahi * blo);
    y = (alo * blo) - err3;
  }

  __host__ __device__
    inline void Square_Tail(real a, real x, real& y, real splitter,
                            real& ahi, real& alo) const
  {
    Split(a, ahi, alo, splitter);
    real err1 = x - (ahi * ahi);
    real err3 = err1 - ((ahi + ahi) * alo);
    y = (alo * alo) - err3;
  }

  __host__ __device__
    inline void Square(real a, real& x, real& y, real splitter,
                       real& ahi, real& alo) const
  {
    x = (real) (a * a);
    Square_Tail(a, x, y, splitter, ahi, alo);
  }

  __host__ __device__
    inline void Two_One_Sum(real a1, real a0, real b,
                            real& x2, real& x1, real& x0, real& _i) const
  {
    Two_Sum(a0, b , _i, x0);
    Two_Sum(a1, _i, x2, x1);
  }

  __host__ __device__
    inline void Two_Two_Sum(real a1, real a0, real b1, real b0,
                            real& x3, real& x2, real& x1, real& x0,
                            real& _j, real& _0, real& _i) const
  {
    Two_One_Sum(a1, a0, b0, _j, _0, x0, _i);
    Two_One_Sum(_j, _0, b1, x3, x2, x1, _i);
  }

  __host__ __device__
    inline void Two_One_Diff(real a1, real a0, real b,
                             real& x2, real& x1, real& x0, real& _i) const
  {
    Two_Diff(a0, b , _i, x0);
    Two_Sum(a1, _i, x2, x1);
  }

  __host__ __device__
    inline void Two_Two_Diff(real a1, real a0, real b1, real b0,
                             real& x3, real& x2, real& x1, real& x0,
                             real& _i, real& _j, real& _0) const
  {
    Two_One_Diff(a1, a0, b0, _j, _0, x0, _i);
    Two_One_Diff(_j, _0, b1, x3, x2, x1, _i);
  }
};

//######################################################################
//
//######################################################################

inline real Predicates::orient2d(real ax, real ay,
                                 real bx, real by,
                                 real cx, real cy,
                                 const real * const pParam) const
{
  real zero = 0.0;
  real detleft, detright, det;
  real detsum, errbound;

  detleft = (ax - cx) * (by - cy);
  detright = (ay - cy) * (bx - cx);
  det = detleft - detright;

  if ( (detleft > zero && detright <= zero) ||
       (detleft < zero && detright >= zero) ||
       detleft == zero) return det;

  if (detleft > zero)
    detsum = detleft + detright;
  else
    detsum = -detleft - detright;

  errbound = pParam[3] * detsum;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  return orient2dadapt(ax, ay, bx, by, cx, cy, detsum, pParam);
}

//######################################################################
//
//######################################################################

inline real Predicates::orient2dadapt(real ax, real ay,
                                      real bx, real by,
                                      real cx, real cy,
                                      real detsum, const real * const pParam) const
{
  real zero = 0.0;
  real acx, acy, bcx, bcy;
  real acxtail, acytail, bcxtail, bcytail;
  real detleft, detright;
  real detlefttail, detrighttail;
  real det, errbound;
  real B[4], C1[8], C2[12], D[16];
  real B3;
  int C1length, C2length, Dlength;
  real u[4];
  real u3;
  real s1, t1;
  real s0, t0;

  real ahi, alo, bhi, blo;
  real _i, _j;
  real _0;

  acx = (real) (ax - cx);
  bcx = (real) (bx - cx);
  acy = (real) (ay - cy);
  bcy = (real) (by - cy);

  Two_Product(acx, bcy, detleft, detlefttail, pParam[0], ahi, alo, bhi, blo);
  Two_Product(acy, bcx, detright, detrighttail, pParam[0], ahi, alo, bhi, blo);

  Two_Two_Diff(detleft, detlefttail, detright, detrighttail,
               B3, B[2], B[1], B[0], _i, _j, _0);
  B[3] = B3;

  det = estimate(4, B);
  errbound = pParam[4] * detsum;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Diff_Tail(ax, cx, acx, acxtail);
  Two_Diff_Tail(bx, cx, bcx, bcxtail);
  Two_Diff_Tail(ay, cy, acy, acytail);
  Two_Diff_Tail(by, cy, bcy, bcytail);

  if ((acxtail == zero) && (acytail == zero)
      && (bcxtail == zero) && (bcytail == zero)) {
    return det;
  }

  errbound = pParam[5] * detsum + pParam[2] * Absolute(det);
  det += (acx * bcytail + bcy * acxtail)
       - (acy * bcxtail + bcx * acytail);
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Product(acxtail, bcy, s1, s0, pParam[0], ahi, alo, bhi, blo);
  Two_Product(acytail, bcx, t1, t0, pParam[0], ahi, alo, bhi, blo);
  Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0], _i, _j, _0);
  u[3] = u3;
  C1length = fast_expansion_sum_zeroelim(4, B, 4, u, C1);

  Two_Product(acx, bcytail, s1, s0, pParam[0], ahi, alo, bhi, blo);
  Two_Product(acy, bcxtail, t1, t0, pParam[0], ahi, alo, bhi, blo);
  Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0], _i, _j, _0);
  u[3] = u3;
  C2length = fast_expansion_sum_zeroelim(C1length, C1, 4, u, C2);

  Two_Product(acxtail, bcytail, s1, s0, pParam[0], ahi, alo, bhi, blo);
  Two_Product(acytail, bcxtail, t1, t0, pParam[0], ahi, alo, bhi, blo);
  Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0], _i, _j, _0);
  u[3] = u3;
  Dlength = fast_expansion_sum_zeroelim(C2length, C2, 4, u, D);

  return(D[Dlength - 1]);
}

//######################################################################
//
//######################################################################

inline real Predicates::incircle(real ax, real ay,
                                 real bx, real by,
                                 real cx, real cy,
                                 real dx, real dy,
                                 const real * const pParam) const
{
  real adx, bdx, cdx, ady, bdy, cdy;
  real bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
  real alift, blift, clift;
  real det;
  real permanent, errbound;

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

  permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * alift
            + (Absolute(cdxady) + Absolute(adxcdy)) * blift
            + (Absolute(adxbdy) + Absolute(bdxady)) * clift;
  errbound = pParam[9] * permanent;
  if ((det > errbound) || (-det > errbound)) {
    return det;
  }

  return incircleadapt(ax, ay, bx, by, cx, cy, dx, dy, permanent, pParam);
}

//######################################################################
//
//######################################################################

inline real Predicates::incircleadapt(real ax, real ay,
                                      real bx, real by,
                                      real cx, real cy,
                                      real dx, real dy,
                                      real permanent,
                                      const real * const pParam) const
{
  real zero = (real) 0.0;
  real two = (real) 2.0;

  real adx, bdx, cdx, ady, bdy, cdy;
  real det, errbound;

  real bdxcdy1, cdxbdy1, cdxady1, adxcdy1, adxbdy1, bdxady1;
  real bdxcdy0, cdxbdy0, cdxady0, adxcdy0, adxbdy0, bdxady0;
  real bc[4], ca[4], ab[4];
  real bc3, ca3, ab3;
  real axbc[8], axxbc[16], aybc[8], ayybc[16], adet[32];
  int axbclen, axxbclen, aybclen, ayybclen, alen;
  real bxca[8], bxxca[16], byca[8], byyca[16], bdet[32];
  int bxcalen, bxxcalen, bycalen, byycalen, blen;
  real cxab[8], cxxab[16], cyab[8], cyyab[16], cdet[32];
  int cxablen, cxxablen, cyablen, cyyablen, clen;
  real abdet[64];
  int ablen;
  real fin1[1152], fin2[1152];
  real *finnow, *finother, *finswap;
  int finlength;

  real adxtail, bdxtail, cdxtail, adytail, bdytail, cdytail;
  real adxadx1, adyady1, bdxbdx1, bdybdy1, cdxcdx1, cdycdy1;
  real adxadx0, adyady0, bdxbdx0, bdybdy0, cdxcdx0, cdycdy0;
  real aa[4], bb[4], cc[4];
  real aa3, bb3, cc3;
  real ti1, tj1;
  real ti0, tj0;
  real u[4], v[4];
  real u3, v3;
  real temp8[8], temp16a[16], temp16b[16], temp16c[16];
  real temp32a[32], temp32b[32], temp48[48], temp64[64];
  int temp8len, temp16alen, temp16blen, temp16clen;
  int temp32alen, temp32blen, temp48len, temp64len;
  real axtbb[8], axtcc[8], aytbb[8], aytcc[8];
  int axtbblen, axtcclen, aytbblen, aytcclen;
  real bxtaa[8], bxtcc[8], bytaa[8], bytcc[8];
  int bxtaalen, bxtcclen, bytaalen, bytcclen;
  real cxtaa[8], cxtbb[8], cytaa[8], cytbb[8];
  int cxtaalen, cxtbblen, cytaalen, cytbblen;
  real axtbc[8], aytbc[8], bxtca[8], bytca[8], cxtab[8], cytab[8];
  int axtbclen = 0, aytbclen = 0,
    bxtcalen = 0, bytcalen = 0,
    cxtablen = 0, cytablen = 0;
  real axtbct[16], aytbct[16], bxtcat[16], bytcat[16], cxtabt[16], cytabt[16];
  int axtbctlen, aytbctlen, bxtcatlen, bytcatlen, cxtabtlen, cytabtlen;
  real axtbctt[8], aytbctt[8], bxtcatt[8];
  real bytcatt[8], cxtabtt[8], cytabtt[8];
  int axtbcttlen, aytbcttlen, bxtcattlen, bytcattlen, cxtabttlen, cytabttlen;
  real abt[8], bct[8], cat[8];
  int abtlen, bctlen, catlen;
  real abtt[4], bctt[4], catt[4];
  int abttlen, bcttlen, cattlen;
  real abtt3, bctt3, catt3;
  real negate;

  real ahi, alo, bhi, blo;
  real _i, _j;
  real _0;

  adx = (real) (ax - dx);
  bdx = (real) (bx - dx);
  cdx = (real) (cx - dx);
  ady = (real) (ay - dy);
  bdy = (real) (by - dy);
  cdy = (real) (cy - dy);

  Two_Product(bdx, cdy, bdxcdy1, bdxcdy0, pParam[0], ahi, alo, bhi, blo);
  Two_Product(cdx, bdy, cdxbdy1, cdxbdy0, pParam[0], ahi, alo, bhi, blo);
  Two_Two_Diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0, bc3, bc[2], bc[1], bc[0],
               _i, _j, _0);
  bc[3] = bc3;
  axbclen = scale_expansion_zeroelim(4, bc, adx, axbc, pParam);
  axxbclen = scale_expansion_zeroelim(axbclen, axbc, adx, axxbc, pParam);
  aybclen = scale_expansion_zeroelim(4, bc, ady, aybc, pParam);
  ayybclen = scale_expansion_zeroelim(aybclen, aybc, ady, ayybc, pParam);
  alen = fast_expansion_sum_zeroelim(axxbclen, axxbc, ayybclen, ayybc, adet);

  Two_Product(cdx, ady, cdxady1, cdxady0, pParam[0], ahi, alo, bhi, blo);
  Two_Product(adx, cdy, adxcdy1, adxcdy0, pParam[0], ahi, alo, bhi, blo);
  Two_Two_Diff(cdxady1, cdxady0, adxcdy1, adxcdy0, ca3, ca[2], ca[1], ca[0],
               _i, _j, _0);
  ca[3] = ca3;
  bxcalen = scale_expansion_zeroelim(4, ca, bdx, bxca, pParam);
  bxxcalen = scale_expansion_zeroelim(bxcalen, bxca, bdx, bxxca, pParam);
  bycalen = scale_expansion_zeroelim(4, ca, bdy, byca, pParam);
  byycalen = scale_expansion_zeroelim(bycalen, byca, bdy, byyca, pParam);
  blen = fast_expansion_sum_zeroelim(bxxcalen, bxxca, byycalen, byyca, bdet);

  Two_Product(adx, bdy, adxbdy1, adxbdy0, pParam[0], ahi, alo, bhi, blo);
  Two_Product(bdx, ady, bdxady1, bdxady0, pParam[0], ahi, alo, bhi, blo);
  Two_Two_Diff(adxbdy1, adxbdy0, bdxady1, bdxady0, ab3, ab[2], ab[1], ab[0],
               _i, _j, _0);
  ab[3] = ab3;
  cxablen = scale_expansion_zeroelim(4, ab, cdx, cxab, pParam);
  cxxablen = scale_expansion_zeroelim(cxablen, cxab, cdx, cxxab, pParam);
  cyablen = scale_expansion_zeroelim(4, ab, cdy, cyab, pParam);
  cyyablen = scale_expansion_zeroelim(cyablen, cyab, cdy, cyyab, pParam);
  clen = fast_expansion_sum_zeroelim(cxxablen, cxxab, cyyablen, cyyab, cdet);

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  finlength = fast_expansion_sum_zeroelim(ablen, abdet, clen, cdet, fin1);

  det = estimate(finlength, fin1);
  errbound = pParam[10] * permanent;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  Two_Diff_Tail(ax, dx, adx, adxtail);
  Two_Diff_Tail(ay, dy, ady, adytail);
  Two_Diff_Tail(bx, dx, bdx, bdxtail);
  Two_Diff_Tail(by, dy, bdy, bdytail);
  Two_Diff_Tail(cx, dx, cdx, cdxtail);
  Two_Diff_Tail(cy, dy, cdy, cdytail);
  if ((adxtail == zero) && (bdxtail == zero) && (cdxtail == zero)
      && (adytail == zero) && (bdytail == zero) && (cdytail == zero)) {
    return det;
  }

  errbound = pParam[11] * permanent + pParam[2] * Absolute(det);
  det += ((adx * adx + ady * ady) * ((bdx * cdytail + cdy * bdxtail)
                                     - (bdy * cdxtail + cdx * bdytail))
          + two * (adx * adxtail + ady * adytail) * (bdx * cdy - bdy * cdx))
       + ((bdx * bdx + bdy * bdy) * ((cdx * adytail + ady * cdxtail)
                                     - (cdy * adxtail + adx * cdytail))
          + two * (bdx * bdxtail + bdy * bdytail) * (cdx * ady - cdy * adx))
       + ((cdx * cdx + cdy * cdy) * ((adx * bdytail + bdy * adxtail)
                                     - (ady * bdxtail + bdx * adytail))
          + two * (cdx * cdxtail + cdy * cdytail) * (adx * bdy - ady * bdx));
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  finnow = fin1;
  finother = fin2;

  if ((bdxtail != zero) || (bdytail != zero)
      || (cdxtail != zero) || (cdytail != zero)) {
    Square(adx, adxadx1, adxadx0, pParam[0], ahi, alo);
    Square(ady, adyady1, adyady0, pParam[0], ahi, alo);
    Two_Two_Sum(adxadx1, adxadx0, adyady1, adyady0,
                aa3, aa[2], aa[1], aa[0], _j, _0, _i);
    aa[3] = aa3;
  }
  if ((cdxtail != zero) || (cdytail != zero)
      || (adxtail != zero) || (adytail != zero)) {
    Square(bdx, bdxbdx1, bdxbdx0, pParam[0], ahi, alo);
    Square(bdy, bdybdy1, bdybdy0, pParam[0], ahi, alo);
    Two_Two_Sum(bdxbdx1, bdxbdx0, bdybdy1,
                bdybdy0, bb3, bb[2], bb[1], bb[0], _j, _0, _i);
    bb[3] = bb3;
  }
  if ((adxtail != zero) || (adytail != zero)
      || (bdxtail != zero) || (bdytail != zero)) {
    Square(cdx, cdxcdx1, cdxcdx0, pParam[0], ahi, alo);
    Square(cdy, cdycdy1, cdycdy0, pParam[0], ahi, alo);
    Two_Two_Sum(cdxcdx1, cdxcdx0, cdycdy1, cdycdy0,
                cc3, cc[2], cc[1], cc[0], _j, _0, _i);
    cc[3] = cc3;
  }

  if (adxtail != zero) {
    axtbclen = scale_expansion_zeroelim(4, bc, adxtail, axtbc, pParam);
    temp16alen = scale_expansion_zeroelim(axtbclen, axtbc, two * adx,
                                          temp16a, pParam);

    axtcclen = scale_expansion_zeroelim(4, cc, adxtail, axtcc, pParam);
    temp16blen = scale_expansion_zeroelim(axtcclen, axtcc, bdy, temp16b,
                                          pParam);

    axtbblen = scale_expansion_zeroelim(4, bb, adxtail, axtbb, pParam);
    temp16clen = scale_expansion_zeroelim(axtbblen, axtbb, -cdy, temp16c,
                                          pParam);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (adytail != zero) {
    aytbclen = scale_expansion_zeroelim(4, bc, adytail, aytbc, pParam);
    temp16alen = scale_expansion_zeroelim(aytbclen, aytbc, two * ady,
                                          temp16a, pParam);

    aytbblen = scale_expansion_zeroelim(4, bb, adytail, aytbb, pParam);
    temp16blen = scale_expansion_zeroelim(aytbblen, aytbb, cdx, temp16b,
                                          pParam);

    aytcclen = scale_expansion_zeroelim(4, cc, adytail, aytcc, pParam);
    temp16clen = scale_expansion_zeroelim(aytcclen, aytcc, -bdx, temp16c,
                                          pParam);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                             temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (bdxtail != zero) {
    bxtcalen = scale_expansion_zeroelim(4, ca, bdxtail, bxtca, pParam);
    temp16alen = scale_expansion_zeroelim(bxtcalen, bxtca, two * bdx,
                                          temp16a, pParam);

    bxtaalen = scale_expansion_zeroelim(4, aa, bdxtail, bxtaa, pParam);
    temp16blen = scale_expansion_zeroelim(bxtaalen, bxtaa, cdy, temp16b,
                                          pParam);

    bxtcclen = scale_expansion_zeroelim(4, cc, bdxtail, bxtcc, pParam);
    temp16clen = scale_expansion_zeroelim(bxtcclen, bxtcc, -ady, temp16c,
                                          pParam);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (bdytail != zero) {
    bytcalen = scale_expansion_zeroelim(4, ca, bdytail, bytca, pParam);
    temp16alen = scale_expansion_zeroelim(bytcalen, bytca, two * bdy,
                                          temp16a, pParam);

    bytcclen = scale_expansion_zeroelim(4, cc, bdytail, bytcc, pParam);
    temp16blen = scale_expansion_zeroelim(bytcclen, bytcc, adx, temp16b,
                                          pParam);

    bytaalen = scale_expansion_zeroelim(4, aa, bdytail, bytaa, pParam);
    temp16clen = scale_expansion_zeroelim(bytaalen, bytaa, -cdx, temp16c,
                                          pParam);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (cdxtail != zero) {
    cxtablen = scale_expansion_zeroelim(4, ab, cdxtail, cxtab, pParam);
    temp16alen = scale_expansion_zeroelim(cxtablen, cxtab, two * cdx,
                                          temp16a, pParam);

    cxtbblen = scale_expansion_zeroelim(4, bb, cdxtail, cxtbb, pParam);
    temp16blen = scale_expansion_zeroelim(cxtbblen, cxtbb, ady, temp16b,
                                          pParam);

    cxtaalen = scale_expansion_zeroelim(4, aa, cdxtail, cxtaa, pParam);
    temp16clen = scale_expansion_zeroelim(cxtaalen, cxtaa, -bdy, temp16c,
                                          pParam);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }
  if (cdytail != zero) {
    cytablen = scale_expansion_zeroelim(4, ab, cdytail, cytab, pParam);
    temp16alen = scale_expansion_zeroelim(cytablen, cytab, two * cdy,
                                          temp16a, pParam);

    cytaalen = scale_expansion_zeroelim(4, aa, cdytail, cytaa, pParam);
    temp16blen = scale_expansion_zeroelim(cytaalen, cytaa, bdx, temp16b,
                                          pParam);

    cytbblen = scale_expansion_zeroelim(4, bb, cdytail, cytbb, pParam);
    temp16clen = scale_expansion_zeroelim(cytbblen, cytbb, -adx, temp16c,
                                          pParam);

    temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                            temp16blen, temp16b, temp32a);
    temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                            temp32alen, temp32a, temp48);
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                            temp48, finother);
    finswap = finnow; finnow = finother; finother = finswap;
  }

  if ((adxtail != zero) || (adytail != zero)) {
    if ((bdxtail != zero) || (bdytail != zero)
        || (cdxtail != zero) || (cdytail != zero)) {
      Two_Product(bdxtail, cdy, ti1, ti0, pParam[0], ahi, alo, bhi, blo);
      Two_Product(bdx, cdytail, tj1, tj0, pParam[0], ahi, alo, bhi, blo);
      Two_Two_Sum(ti1, ti0, tj1, tj0, u3, u[2], u[1], u[0], _j, _0, _i);
      u[3] = u3;
      negate = -bdy;
      Two_Product(cdxtail, negate, ti1, ti0, pParam[0], ahi, alo, bhi, blo);
      negate = -bdytail;
      Two_Product(cdx, negate, tj1, tj0, pParam[0], ahi, alo, bhi, blo);
      Two_Two_Sum(ti1, ti0, tj1, tj0, v3, v[2], v[1], v[0], _j, _0, _i);
      v[3] = v3;
      bctlen = fast_expansion_sum_zeroelim(4, u, 4, v, bct);

      Two_Product(bdxtail, cdytail, ti1, ti0, pParam[0], ahi, alo, bhi, blo);
      Two_Product(cdxtail, bdytail, tj1, tj0, pParam[0], ahi, alo, bhi, blo);
      Two_Two_Diff(ti1, ti0, tj1, tj0, bctt3, bctt[2], bctt[1], bctt[0],
                   _i, _j, _0);
      bctt[3] = bctt3;
      bcttlen = 4;
    } else {
      bct[0] = zero;
      bctlen = 1;
      bctt[0] = zero;
      bcttlen = 1;
    }

    if (adxtail != zero) {
      temp16alen = scale_expansion_zeroelim(axtbclen, axtbc, adxtail,
                                            temp16a, pParam);
      axtbctlen = scale_expansion_zeroelim(bctlen, bct, adxtail, axtbct,
                                           pParam);
      temp32alen = scale_expansion_zeroelim(axtbctlen, axtbct, two * adx,
                                            temp32a, pParam);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (bdytail != zero) {
        temp8len = scale_expansion_zeroelim(4, cc, adxtail, temp8, pParam);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, bdytail,
                                              temp16a, pParam);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
      if (cdytail != zero) {
        temp8len = scale_expansion_zeroelim(4, bb, -adxtail, temp8, pParam);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, cdytail,
                                              temp16a, pParam);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }

      temp32alen = scale_expansion_zeroelim(axtbctlen, axtbct, adxtail,
                                            temp32a, pParam);
      axtbcttlen = scale_expansion_zeroelim(bcttlen, bctt, adxtail, axtbctt,
                                            pParam);
      temp16alen = scale_expansion_zeroelim(axtbcttlen, axtbctt, two * adx,
                                            temp16a, pParam);
      temp16blen = scale_expansion_zeroelim(axtbcttlen, axtbctt, adxtail,
                                            temp16b, pParam);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
    if (adytail != zero) {
      temp16alen = scale_expansion_zeroelim(aytbclen, aytbc, adytail,
                                            temp16a, pParam);
      aytbctlen = scale_expansion_zeroelim(bctlen, bct, adytail, aytbct,
                                           pParam);
      temp32alen = scale_expansion_zeroelim(aytbctlen, aytbct, two * ady,
                                            temp32a, pParam);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;


      temp32alen = scale_expansion_zeroelim(aytbctlen, aytbct, adytail,
                                            temp32a, pParam);
      aytbcttlen = scale_expansion_zeroelim(bcttlen, bctt, adytail, aytbctt,
                                            pParam);
      temp16alen = scale_expansion_zeroelim(aytbcttlen, aytbctt, two * ady,
                                            temp16a, pParam);
      temp16blen = scale_expansion_zeroelim(aytbcttlen, aytbctt, adytail,
                                            temp16b, pParam);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
  }
  if ((bdxtail != zero) || (bdytail != zero)) {
    if ((cdxtail != zero) || (cdytail != zero)
        || (adxtail != zero) || (adytail != zero)) {
      Two_Product(cdxtail, ady, ti1, ti0, pParam[0], ahi, alo, bhi, blo);
      Two_Product(cdx, adytail, tj1, tj0, pParam[0], ahi, alo, bhi, blo);
      Two_Two_Sum(ti1, ti0, tj1, tj0, u3, u[2], u[1], u[0], _j, _0, _i);
      u[3] = u3;
      negate = -cdy;
      Two_Product(adxtail, negate, ti1, ti0, pParam[0], ahi, alo, bhi, blo);
      negate = -cdytail;
      Two_Product(adx, negate, tj1, tj0, pParam[0], ahi, alo, bhi, blo);
      Two_Two_Sum(ti1, ti0, tj1, tj0, v3, v[2], v[1], v[0], _j, _0, _i);
      v[3] = v3;
      catlen = fast_expansion_sum_zeroelim(4, u, 4, v, cat);

      Two_Product(cdxtail, adytail, ti1, ti0, pParam[0], ahi, alo, bhi, blo);
      Two_Product(adxtail, cdytail, tj1, tj0, pParam[0], ahi, alo, bhi, blo);
      Two_Two_Diff(ti1, ti0, tj1, tj0, catt3, catt[2], catt[1], catt[0],
                   _i, _j, _0);
      catt[3] = catt3;
      cattlen = 4;
    } else {
      cat[0] = zero;
      catlen = 1;
      catt[0] = zero;
      cattlen = 1;
    }

    if (bdxtail != zero) {
      temp16alen = scale_expansion_zeroelim(bxtcalen, bxtca, bdxtail,
                                            temp16a, pParam);
      bxtcatlen = scale_expansion_zeroelim(catlen, cat, bdxtail, bxtcat,
                                           pParam);
      temp32alen = scale_expansion_zeroelim(bxtcatlen, bxtcat, two * bdx,
                                            temp32a, pParam);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (cdytail != zero) {
        temp8len = scale_expansion_zeroelim(4, aa, bdxtail, temp8, pParam);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, cdytail,
                                              temp16a, pParam);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
      if (adytail != zero) {
        temp8len = scale_expansion_zeroelim(4, cc, -bdxtail, temp8, pParam);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, adytail,
                                              temp16a, pParam);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }

      temp32alen = scale_expansion_zeroelim(bxtcatlen, bxtcat, bdxtail,
                                            temp32a, pParam);
      bxtcattlen = scale_expansion_zeroelim(cattlen, catt, bdxtail, bxtcatt,
                                            pParam);
      temp16alen = scale_expansion_zeroelim(bxtcattlen, bxtcatt, two * bdx,
                                            temp16a, pParam);
      temp16blen = scale_expansion_zeroelim(bxtcattlen, bxtcatt, bdxtail,
                                            temp16b, pParam);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
    if (bdytail != zero) {
      temp16alen = scale_expansion_zeroelim(bytcalen, bytca, bdytail,
                                            temp16a, pParam);
      bytcatlen = scale_expansion_zeroelim(catlen, cat, bdytail, bytcat,
                                           pParam);
      temp32alen = scale_expansion_zeroelim(bytcatlen, bytcat, two * bdy,
                                            temp32a, pParam);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;


      temp32alen = scale_expansion_zeroelim(bytcatlen, bytcat, bdytail,
                                            temp32a, pParam);
      bytcattlen = scale_expansion_zeroelim(cattlen, catt, bdytail, bytcatt,
                                            pParam);
      temp16alen = scale_expansion_zeroelim(bytcattlen, bytcatt, two * bdy,
                                            temp16a, pParam);
      temp16blen = scale_expansion_zeroelim(bytcattlen, bytcatt, bdytail,
                                            temp16b, pParam);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
  }
  if ((cdxtail != zero) || (cdytail != zero)) {
    if ((adxtail != zero) || (adytail != zero)
        || (bdxtail != zero) || (bdytail != zero)) {
      Two_Product(adxtail, bdy, ti1, ti0, pParam[0], ahi, alo, bhi, blo);
      Two_Product(adx, bdytail, tj1, tj0, pParam[0], ahi, alo, bhi, blo);
      Two_Two_Sum(ti1, ti0, tj1, tj0, u3, u[2], u[1], u[0], _j, _0, _i);
      u[3] = u3;
      negate = -ady;
      Two_Product(bdxtail, negate, ti1, ti0, pParam[0], ahi, alo, bhi, blo);
      negate = -adytail;
      Two_Product(bdx, negate, tj1, tj0, pParam[0], ahi, alo, bhi, blo);
      Two_Two_Sum(ti1, ti0, tj1, tj0, v3, v[2], v[1], v[0], _j, _0, _i);
      v[3] = v3;
      abtlen = fast_expansion_sum_zeroelim(4, u, 4, v, abt);

      Two_Product(adxtail, bdytail, ti1, ti0, pParam[0], ahi, alo, bhi, blo);
      Two_Product(bdxtail, adytail, tj1, tj0, pParam[0], ahi, alo, bhi, blo);
      Two_Two_Diff(ti1, ti0, tj1, tj0, abtt3, abtt[2], abtt[1], abtt[0],
                   _i, _j, _0);
      abtt[3] = abtt3;
      abttlen = 4;
    } else {
      abt[0] = zero;
      abtlen = 1;
      abtt[0] = zero;
      abttlen = 1;
    }

    if (cdxtail != zero) {
      temp16alen = scale_expansion_zeroelim(cxtablen, cxtab, cdxtail,
                                            temp16a, pParam);
      cxtabtlen = scale_expansion_zeroelim(abtlen, abt, cdxtail, cxtabt,
                                           pParam);
      temp32alen = scale_expansion_zeroelim(cxtabtlen, cxtabt, two * cdx,
                                            temp32a, pParam);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;
      if (adytail != zero) {
        temp8len = scale_expansion_zeroelim(4, bb, cdxtail, temp8, pParam);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, adytail,
                                              temp16a, pParam);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }
      if (bdytail != zero) {
        temp8len = scale_expansion_zeroelim(4, aa, -cdxtail, temp8, pParam);
        temp16alen = scale_expansion_zeroelim(temp8len, temp8, bdytail,
                                              temp16a, pParam);
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                                                temp16a, finother);
        finswap = finnow; finnow = finother; finother = finswap;
      }

      temp32alen = scale_expansion_zeroelim(cxtabtlen, cxtabt, cdxtail,
                                            temp32a, pParam);
      cxtabttlen = scale_expansion_zeroelim(abttlen, abtt, cdxtail, cxtabtt,
                                            pParam);
      temp16alen = scale_expansion_zeroelim(cxtabttlen, cxtabtt, two * cdx,
                                            temp16a, pParam);
      temp16blen = scale_expansion_zeroelim(cxtabttlen, cxtabtt, cdxtail,
                                            temp16b, pParam);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
    if (cdytail != zero) {
      temp16alen = scale_expansion_zeroelim(cytablen, cytab, cdytail,
                                            temp16a, pParam);
      cytabtlen = scale_expansion_zeroelim(abtlen, abt, cdytail, cytabt,
                                           pParam);
      temp32alen = scale_expansion_zeroelim(cytabtlen, cytabt, two * cdy,
                                            temp32a, pParam);
      temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp32alen, temp32a, temp48);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                              temp48, finother);
      finswap = finnow; finnow = finother; finother = finswap;


      temp32alen = scale_expansion_zeroelim(cytabtlen, cytabt, cdytail,
                                            temp32a, pParam);
      cytabttlen = scale_expansion_zeroelim(abttlen, abtt, cdytail, cytabtt,
                                            pParam);
      temp16alen = scale_expansion_zeroelim(cytabttlen, cytabtt, two * cdy,
                                            temp16a, pParam);
      temp16blen = scale_expansion_zeroelim(cytabttlen, cytabtt, cdytail,
                                            temp16b, pParam);
      temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                              temp16blen, temp16b, temp32b);
      temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                              temp32blen, temp32b, temp64);
      finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                                              temp64, finother);
      finswap = finnow; finnow = finother; finother = finswap;
    }
  }

  return finnow[finlength - 1];
}

}  // namespace astrix

#endif  // ASTRIX_PREDICATES_H
