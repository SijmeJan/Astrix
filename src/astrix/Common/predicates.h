/*! \file predicates.h 
\brief Header file containing Predicates class definition
*/
#ifndef PREDICATES_H
#define PREDICATES_H

#define INLINE_ORIENT 1
#define INLINE_INCIRCLE 1
#define FAST_ORIENT 1
#define FAST_INCIRCLE 1

//#include "./essa.h"

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
  Predicates(Device *device);
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
		real *pParam) const;

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
		real *pParam) const;

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
		     real *pParam) const;
  __host__ __device__
  real orient2dadapt(real ax, real ay,
		     real bx, real by,
		     real cx, real cy,
		     real detsum,
		     real *pParam) const;

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
					real *pParam) const
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
    Two_Sum( a1, _i, x2, x1);
  }
  
  __host__ __device__
    inline void Two_Two_Diff(real a1, real a0, real b1, real b0,
			     real& x3, real& x2, real& x1, real& x0,
			     real& _i, real& _j, real& _0) const
  {
    Two_One_Diff(a1, a0, b0, _j, _0, x0, _i); 
    Two_One_Diff(_j, _0, b1, x3, x2, x1, _i);
  }

  /*
  __host__ __device__
    inline void Fast_Two_Sum(real a, real b, real& x, real& y);
  __host__ __device__
    inline void Fast_Two_Diff_Tail(real a, real b, real x, real& y);
  __host__ __device__
    inline void Fast_Two_Diff(real a, real b, real& x, real& y);
  __host__ __device__
    inline void Two_Sum_Tail(real a, real b, real x, real& y);
  __host__ __device__
  inline void Two_Sum(real a, real b, real& x, real& y);
  __host__ __device__
  inline void Two_Diff_Tail(real a, real b, real x, real& y);
  __host__ __device__
  inline void Two_Diff(real a, real b, real& x, real& y);
  __host__ __device__
  inline void Split(real a, real& ahi, real& alo, real splitter);
  __host__ __device__
  inline void Two_Product_Tail(real a, real b, real x, real& y, real splitter,
			       real& ahi, real& alo,
			       real& bhi, real& blo);
  __host__ __device__
  inline void Two_Product(real a, real b, real& x, real& y, real splitter,
			  real& ahi, real& alo,
			  real& bhi, real& blo); 
  // Two_Product_Presplit() is Two_Product() where one of the inputs has       
  //   already been split.  Avoids redundant splitting.                        
  __host__ __device__
  inline void Two_Product_Presplit(real a, real b,
				   real bhi, real blo,
				   real& x, real& y, real splitter,
				   real& ahi, real& alo);

  // Two_Product_2Presplit() is Two_Product() where both of the inputs have    
  //   already been split.  Avoids redundant splitting.                        
  __host__ __device__
  inline void Two_Product_2Presplit(real a, real ahi, real alo,
				    real b, real bhi, real blo,
				    real& x, real& y);

  __host__ __device__
  inline void Square_Tail(real a, real x, real& y, real splitter,
			  real& ahi, real& alo);
  __host__ __device__
  inline void Square(real a, real& x, real& y, real splitter,
    real& ahi, real& alo);	

  __host__ __device__
  inline void Two_One_Sum(real a1, real a0, real b,
    real& x2, real& x1, real& x0, real& _i);
  
  __host__ __device__
  inline void Two_Two_Sum(real a1, real a0, real b1, real b0,
    real& x3, real& x2, real& x1, real& x0, real& _j, real& _0, real& _i);

  __host__ __device__
  inline void Two_One_Diff(real a1, real a0, real b,
    real& x2, real& x1, real& x0, real& _i);

  __host__ __device__
  inline void Two_Two_Diff(real a1, real a0, real b1, real b0,
    real& x3, real& x2, real& x1, real& x0, real& _i, real& _j, real& _0);
  */
};

#if INLINE_ORIENT==1
//######################################################################
// 
//######################################################################

inline real Predicates::orient2d(real ax, real ay,
				 real bx, real by,
				 real cx, real cy,
				 real *pParam) const
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

  /*
  if (detleft > zero) {
    if (detright <= zero) {
      return det;
    } else {
      detsum = detleft + detright;
    }
  } else if (detleft < zero) {
    if (detright >= zero) {
      return det;
    } else {
      detsum = -detleft - detright;
    }
  } else {
    return det;
  }
  */
  
  errbound = pParam[3] * detsum;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

#if (USE_DOUBLE == -1 && FAST_ORIENT == 1)

  real ss = orient2dadapt(ax, ay, bx, by, cx, cy, detsum, pParam);

  /*
#ifndef __CUDA_ARCH__
  double dax = (double) ax;
  double day = (double) ay;
  double dbx = (double) bx;
  double dby = (double) by;
  double dcx = (double) cx;
  double dcy = (double) cy;
  
  double ddetleft  = (dax - dcx)*(dby - dcy);
  double ddetright = (day - dcy)*(dbx - dcx);
  double ddet = ddetleft - ddetright;

  if (((double)ss)*ddet < 0.0 ||
      (ss == 0.0f && ddet != 0.0) ||
      (ss != 0.0f && ddet == 0.0)) {

    double r[6];
    r[0] = (double) ax * (double) by;
    r[1] = (double) bx * (double) cy;
    r[2] = (double) cx * (double) ay;
    r[3] = -(double) ax * (double) cy;
    r[4] = -(double) bx * (double) ay;
    r[5] = -(double) cx * (double) by;
    real s = ExactSignSum<double, 6>(r)*errbound;

    std::cout << std::endl
	      << "fp64 and orient2dadapt give different answers: "
	      << "orient2d: " << ss << ", fp64: " << ddet
	      << ", exactSign: " << s << std::endl;
  }
#endif
  */
  
  return ss;

#else
  
  return orient2dadapt(ax, ay, bx, by, cx, cy, detsum, pParam);

#endif
}

//######################################################################
// 
//######################################################################

inline real Predicates::orient2dadapt(real ax, real ay,
				      real bx, real by,
				      real cx, real cy,
				      real detsum, real *pParam) const
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

  // real bvirt;
  //real avirt, bround, around;
  // real c;
  // real abig;
  real ahi, alo, bhi, blo;
  //real err1, err2, err3;
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

#endif

/*
#define Fast_Two_Sum_Tail(a, b, x, y)		\
bvirt = x - a;					\
y = b - bvirt

#define Fast_Two_Sum(a, b, x, y)		\
x = (REAL) (a + b);				\
Fast_Two_Sum_Tail(a, b, x, y)

#define Fast_Two_Diff_Tail(a, b, x, y)		\
bvirt = a - x;					\
y = bvirt - b

#define Fast_Two_Diff(a, b, x, y)		\
x = (REAL) (a - b);				\
Fast_Two_Diff_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y)		\
bvirt = (REAL) (x - a);				\
avirt = x - bvirt;				\
bround = b - bvirt;				\
around = a - avirt;				\
y = around + bround

#define Two_Sum(a, b, x, y)			\
x = (REAL) (a + b);				\
Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y)		\
bvirt = (REAL) (a - x);				\
avirt = x + bvirt;				\
bround = bvirt - b;				\
around = a - avirt;				\
y = around + bround

#define Two_Diff(a, b, x, y)			\
x = (REAL) (a - b);				\
Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo, splitter)		\
c = (REAL) (splitter * a);			\
abig = (REAL) (c - a);				\
ahi = c - abig;					\
alo = a - ahi

#define Two_Product_Tail(a, b, x, y, splitter)		\
Split(a, ahi, alo, splitter);				\
Split(b, bhi, blo, splitter);				\
err1 = x - (ahi * bhi);					\
err2 = err1 - (alo * bhi);				\
err3 = err2 - (ahi * blo);				\
y = (alo * blo) - err3

#define Two_Product(a, b, x, y, splitter)	\
x = (REAL) (a * b);				\
Two_Product_Tail(a, b, x, y, splitter)

// Two_Product_Presplit() is Two_Product() where one of the inputs has       
//   already been split.  Avoids redundant splitting.                        

#define Two_Product_Presplit(a, b, bhi, blo, x, y, splitter)	\
x = (REAL) (a * b);						\
Split(a, ahi, alo, splitter);					\
err1 = x - (ahi * bhi);						\
err2 = err1 - (alo * bhi);					\
err3 = err2 - (ahi * blo);					\
y = (alo * blo) - err3

// Two_Product_2Presplit() is Two_Product() where both of the inputs have    
//   already been split.  Avoids redundant splitting.                        

#define Two_Product_2Presplit(a, ahi, alo, b, bhi, blo, x, y) \
x = (REAL) (a * b);						\
err1 = x - (ahi * bhi);					\
err2 = err1 - (alo * bhi);					\
err3 = err2 - (ahi * blo);					\
y = (alo * blo) - err3

// Square() can be done more quickly than Two_Product().                     

#define Square_Tail(a, x, y, splitter)			\
Split(a, ahi, alo, splitter);				\
err1 = x - (ahi * ahi);					\
err3 = err1 - ((ahi + ahi) * alo);				\
y = (alo * alo) - err3

#define Square(a, x, y, splitter)		\
x = (REAL) (a * a);				\
Square_Tail(a, x, y, splitter)

// Macros for summing expansions of various fixed lengths.  These are all    
//   unrolled versions of Expansion_Sum().                                   

#define Two_One_Sum(a1, a0, b, x2, x1, x0)	\
Two_Sum(a0, b , _i, x0);			\
Two_Sum(a1, _i, x2, x1)

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
Two_Diff(a0, b , _i, x0);			\
Two_Sum( a1, _i, x2, x1)

#define Two_Two_Sum(a1, a0, b1, b0, x3, x2, x1, x0) \
Two_One_Sum(a1, a0, b0, _j, _0, x0);		\
Two_One_Sum(_j, _0, b1, x3, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0)	\
Two_One_Diff(a1, a0, b0, _j, _0, x0);			\
Two_One_Diff(_j, _0, b1, x3, x2, x1)

#define Four_One_Sum(a3, a2, a1, a0, b, x4, x3, x2, x1, x0) \
Two_One_Sum(a1, a0, b , _j, x1, x0); \
Two_One_Sum(a3, a2, _j, x4, x3, x2)

#define Four_Two_Sum(a3, a2, a1, a0, b1, b0, x5, x4, x3, x2, x1, x0) \
Four_One_Sum(a3, a2, a1, a0, b0, _k, _2, _1, _0, x0); \
Four_One_Sum(_k, _2, _1, _0, b1, x5, x4, x3, x2, x1)

#define Four_Four_Sum(a3, a2, a1, a0, b4, b3, b1, b0, x7, x6, x5, x4, x3, x2, \
                      x1, x0) \
Four_Two_Sum(a3, a2, a1, a0, b1, b0, _l, _2, _1, _0, x1, x0); \
Four_Two_Sum(_l, _2, _1, _0, b4, b3, x7, x6, x5, x4, x3, x2)

#define Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b, x8, x7, x6, x5, x4, \
                      x3, x2, x1, x0) \
Four_One_Sum(a3, a2, a1, a0, b , _j, x3, x2, x1, x0); \
Four_One_Sum(a7, a6, a5, a4, _j, x8, x7, x6, x5, x4)

#define Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0, x9, x8, x7, \
                      x6, x5, x4, x3, x2, x1, x0) \
Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b0, _k, _6, _5, _4, _3, _2, \
              _1, _0, x0); \
Eight_One_Sum(_k, _6, _5, _4, _3, _2, _1, _0, b1, x9, x8, x7, x6, x5, x4, \
              x3, x2, x1)

#define Eight_Four_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b4, b3, b1, b0, x11, \
                       x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0) \
Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0, _l, _6, _5, _4, _3, \
              _2, _1, _0, x1, x0); \
Eight_Two_Sum(_l, _6, _5, _4, _3, _2, _1, _0, b4, b3, x11, x10, x9, x8, \
              x7, x6, x5, x4, x3, x2)

// Macros for multiplying expansions of various fixed lengths.               

#define Two_One_Product(a1, a0, b, x3, x2, x1, x0, splitter)	\
Split(b, bhi, blo, splitter);				     	\
Two_Product_Presplit(a0, b, bhi, blo, _i, x0, splitter);      \
Two_Product_Presplit(a1, b, bhi, blo, _j, _0, splitter);    	\
Two_Sum(_i, _0, _k, x1); \
Fast_Two_Sum(_j, _k, x3, x2)

#define Four_One_Product(a3, a2, a1, a0, b, x7, x6, x5, x4, x3, x2, x1, x0, splitter) \
Split(b, bhi, blo, splitter);				      \
Two_Product_Presplit(a0, b, bhi, blo, _i, x0, splitter);	      \
Two_Product_Presplit(a1, b, bhi, blo, _j, _0, splitter);		      \
Two_Sum(_i, _0, _k, x1); \
Fast_Two_Sum(_j, _k, _i, x2); \
Two_Product_Presplit(a2, b, bhi, blo, _j, _0, splitter);	\
Two_Sum(_i, _0, _k, x3); \
Fast_Two_Sum(_j, _k, _i, x4); \
Two_Product_Presplit(a3, b, bhi, blo, _j, _0, splitter);	\
Two_Sum(_i, _0, _k, x5); \
Fast_Two_Sum(_j, _k, x7, x6)

#define Two_Two_Product(a1, a0, b1, b0, x7, x6, x5, x4, x3, x2, x1, x0, splitter) \
Split(a0, a0hi, a0lo, splitter);			     \
Split(b0, bhi, blo, splitter);					       \
Two_Product_2Presplit(a0, a0hi, a0lo, b0, bhi, blo, _i, x0); \
Split(a1, a1hi, a1lo, splitter);				       \
Two_Product_2Presplit(a1, a1hi, a1lo, b0, bhi, blo, _j, _0); \
Two_Sum(_i, _0, _k, _1); \
Fast_Two_Sum(_j, _k, _l, _2); \
Split(b1, bhi, blo, splitter);					       \
Two_Product_2Presplit(a0, a0hi, a0lo, b1, bhi, blo, _i, _0); \
Two_Sum(_1, _0, _k, x1); \
Two_Sum(_2, _k, _j, _1); \
Two_Sum(_l, _j, _m, _2); \
Two_Product_2Presplit(a1, a1hi, a1lo, b1, bhi, blo, _j, _0); \
Two_Sum(_i, _0, _n, _0); \
Two_Sum(_1, _0, _i, x2); \
Two_Sum(_2, _i, _k, _1); \
Two_Sum(_m, _k, _l, _2); \
Two_Sum(_j, _n, _k, _0); \
Two_Sum(_1, _0, _j, x3); \
Two_Sum(_2, _j, _i, _1); \
Two_Sum(_l, _i, _m, _2); \
Two_Sum(_1, _k, _i, x4); \
Two_Sum(_2, _i, _k, x5); \
Two_Sum(_m, _k, x7, x6)

// An expansion of length two can be squared more quickly than finding the   
//   product of two different expansions of length two, and the result is    
//   guaranteed to have no more than six (rather than eight) components.     

#define Two_Square(a1, a0, x5, x4, x3, x2, x1, x0, splitter)	\
  Square(a0, _j, x0, splitter);						\
  _0 = a0 + a0; \
  Two_Product(a1, _0, _k, _1, splitter);	       \
  Two_One_Sum(_k, _1, _j, _l, _2, x1); \
  Square(a1, _j, _1, splitter);				\
  Two_Two_Sum(_j, _1, _l, _2, x5, x4, x3, x2)
*/

#if INLINE_INCIRCLE==1
//######################################################################
// 
//######################################################################

//__host__ __device__
inline real Predicates::incircle(real ax, real ay,
				 real bx, real by,
				 real cx, real cy,
				 real dx, real dy,
				 real *pParam) const
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
    //#ifndef __CUDA_ARCH__
    //nPredicatesFast++;
    //#endif
    return det;
  }

  //#ifndef __CUDA_ARCH__
  //nPredicatesSlow++;
  //#endif

#if FAST_INCIRCLE==1 && USE_DOUBLE==-1
  /*
  double r[192];
  SplitProduct(ay, bx, bx, cx,  r[0],  r[1],  r[2],  r[3]);
  SplitProduct(-ax, ax, by, cx, r[4],  r[5],  r[6],  r[7]);
  SplitProduct(-ay, ay, by, cx, r[8],  r[9],  r[10], r[11]);
  SplitProduct(ay, by, by, cx,  r[12], r[13], r[14], r[15]);
  SplitProduct(-ay, bx, cx, cx, r[16], r[17], r[18], r[19]);
  SplitProduct(ax, by, cx, cx,  r[20], r[21], r[22], r[23]);
  SplitProduct(ax, ax, bx, cy,  r[24], r[25], r[26], r[27]);
  SplitProduct(ay, ay, bx, cy,  r[28], r[29], r[30], r[31]);
  
  SplitProduct(-ax, bx, bx, cy, r[32], r[33], r[34], r[35]);
  SplitProduct(-ax, by, by, cy, r[36], r[37], r[38], r[39]);
  SplitProduct(-ay, bx, cy, cy, r[40], r[41], r[42], r[43]);
  SplitProduct(ax, by, cy, cy,  r[44], r[45], r[46], r[47]);
  SplitProduct(-ay, bx, bx, dx, r[48], r[49], r[50], r[51]);
  SplitProduct(ax, ax, by, dx,  r[52], r[53], r[54], r[55]);
  SplitProduct(ay, ay, by, dx,  r[56], r[57], r[58], r[59]);
  SplitProduct(-ay, by, by, dx, r[60], r[61], r[62], r[63]);

  SplitProduct(ay, cx, cx, dx,  r[64], r[65], r[66], r[67]);
  SplitProduct(-by, cx, cx, dx, r[68], r[69], r[70], r[71]);
  SplitProduct(-ax, ax, cy, dx, r[72], r[73], r[74], r[75]);
  SplitProduct(-ay, ay, cy, dx, r[76], r[77], r[78], r[79]);
  SplitProduct(bx, bx, cy, dx,  r[80], r[81], r[82], r[83]);  
  SplitProduct(by, by, cy, dx,  r[84], r[85], r[86], r[87]);
  SplitProduct(ay, cy, cy, dx,  r[88], r[89], r[90], r[91]);
  SplitProduct(-by, cy, cy, dx, r[92], r[93], r[94], r[95]);
  
  SplitProduct(ay, bx, dx, dx,  r[96],  r[97],  r[98],  r[99]);
  SplitProduct(-ax, by, dx, dx, r[100], r[101], r[102], r[103]);
  SplitProduct(-ay, cx, dx, dx, r[104], r[105], r[106], r[107]);
  SplitProduct(by, cx, dx, dx,  r[108], r[109], r[110], r[111]);
  SplitProduct(ax, cy, dx, dx,  r[112], r[113], r[114], r[115]);
  SplitProduct(-bx, cy, dx, dx, r[116], r[117], r[118], r[119]);
  SplitProduct(-ax, ax, bx, dy, r[120], r[121], r[122], r[123]);
  SplitProduct(-ay, ay, bx, dy, r[124], r[125], r[126], r[127]);
  
  SplitProduct(ax, bx, bx, dy,  r[128], r[129], r[130], r[131]);
  SplitProduct(ax, by, by, dy,  r[132], r[133], r[134], r[135]);
  SplitProduct(ax, ax, cx, dy,  r[136], r[137], r[138], r[139]);
  SplitProduct(ay, ay, cx, dy,  r[140], r[141], r[142], r[143]);
  SplitProduct(-bx, bx, cx, dy, r[144], r[145], r[146], r[147]);
  SplitProduct(-by, by, cx, dy, r[148], r[149], r[150], r[151]);
  SplitProduct(-ax, cx, cx, dy, r[152], r[153], r[154], r[155]);
  SplitProduct(bx, cx, cx, dy,  r[156], r[157], r[158], r[159]);
  
  SplitProduct(-ax, cy, cy, dy, r[160], r[161], r[162], r[163]);  
  SplitProduct(bx, cy, cy, dy,  r[164], r[165], r[166], r[167]);
  SplitProduct(ay, bx, dy, dy,  r[168], r[169], r[170], r[171]);
  SplitProduct(-ax, by, dy, dy, r[172], r[173], r[174], r[175]);
  SplitProduct(-ay, cx, dy, dy, r[176], r[177], r[178], r[179]);
  SplitProduct(by, cx, dy, dy,  r[180], r[181], r[182], r[183]);  
  SplitProduct(ax, cy, dy, dy,  r[184], r[185], r[186], r[187]);
  SplitProduct(-bx, cy, dy, dy, r[188], r[189], r[190], r[191]);
  
  real ss = ExactSignSum<double, 192>(r);
  */
  real ss = incircleadapt(ax, ay, bx, by, cx, cy, dx, dy, permanent, pParam);

  /*
#ifndef __CUDA_ARCH__
  double dadx = (double) ax - (double) dx;
  double dbdx = (double) bx - (double) dx;
  double dcdx = (double) cx - (double) dx;
  double dady = (double) ay - (double) dy;
  double dbdy = (double) by - (double) dy;
  double dcdy = (double) cy - (double) dy;

  double dbdxcdy = dbdx * dcdy;
  double dcdxbdy = dcdx * dbdy;
  double dalift = dadx * dadx + dady * dady;

  double dcdxady = dcdx * dady;
  double dadxcdy = dadx * dcdy;
  double dblift = dbdx * dbdx + dbdy * dbdy;

  double dadxbdy = dadx * dbdy;
  double dbdxady = dbdx * dady;
  double dclift = dcdx * dcdx + dcdy * dcdy;

  double ddet =
    dalift * (dbdxcdy - dcdxbdy) +
    dblift * (dcdxady - dadxcdy) +
    dclift * (dadxbdy - dbdxady);

  if (((double)ss)*ddet < 0.0 ||
      (ss == 0.0f && ddet != 0.0) ||
      (ss != 0.0f && ddet == 0.0)) {

    double r[192];
    SplitProduct(ay, bx, bx, cx,  r[0],  r[1],  r[2],  r[3]);
    SplitProduct(-ax, ax, by, cx, r[4],  r[5],  r[6],  r[7]);
    SplitProduct(-ay, ay, by, cx, r[8],  r[9],  r[10], r[11]);
    SplitProduct(ay, by, by, cx,  r[12], r[13], r[14], r[15]);
    SplitProduct(-ay, bx, cx, cx, r[16], r[17], r[18], r[19]);
    SplitProduct(ax, by, cx, cx,  r[20], r[21], r[22], r[23]);
    SplitProduct(ax, ax, bx, cy,  r[24], r[25], r[26], r[27]);
    SplitProduct(ay, ay, bx, cy,  r[28], r[29], r[30], r[31]);
    
    SplitProduct(-ax, bx, bx, cy, r[32], r[33], r[34], r[35]);
    SplitProduct(-ax, by, by, cy, r[36], r[37], r[38], r[39]);
    SplitProduct(-ay, bx, cy, cy, r[40], r[41], r[42], r[43]);
    SplitProduct(ax, by, cy, cy,  r[44], r[45], r[46], r[47]);
    SplitProduct(-ay, bx, bx, dx, r[48], r[49], r[50], r[51]);
    SplitProduct(ax, ax, by, dx,  r[52], r[53], r[54], r[55]);
    SplitProduct(ay, ay, by, dx,  r[56], r[57], r[58], r[59]);
    SplitProduct(-ay, by, by, dx, r[60], r[61], r[62], r[63]);

    SplitProduct(ay, cx, cx, dx,  r[64], r[65], r[66], r[67]);
    SplitProduct(-by, cx, cx, dx, r[68], r[69], r[70], r[71]);
    SplitProduct(-ax, ax, cy, dx, r[72], r[73], r[74], r[75]);
    SplitProduct(-ay, ay, cy, dx, r[76], r[77], r[78], r[79]);
    SplitProduct(bx, bx, cy, dx,  r[80], r[81], r[82], r[83]);  
    SplitProduct(by, by, cy, dx,  r[84], r[85], r[86], r[87]);
    SplitProduct(ay, cy, cy, dx,  r[88], r[89], r[90], r[91]);
    SplitProduct(-by, cy, cy, dx, r[92], r[93], r[94], r[95]);

    SplitProduct(ay, bx, dx, dx,  r[96],  r[97],  r[98],  r[99]);
    SplitProduct(-ax, by, dx, dx, r[100], r[101], r[102], r[103]);
    SplitProduct(-ay, cx, dx, dx, r[104], r[105], r[106], r[107]);
    SplitProduct(by, cx, dx, dx,  r[108], r[109], r[110], r[111]);
    SplitProduct(ax, cy, dx, dx,  r[112], r[113], r[114], r[115]);
    SplitProduct(-bx, cy, dx, dx, r[116], r[117], r[118], r[119]);
    SplitProduct(-ax, ax, bx, dy, r[120], r[121], r[122], r[123]);
    SplitProduct(-ay, ay, bx, dy, r[124], r[125], r[126], r[127]);

    SplitProduct(ax, bx, bx, dy,  r[128], r[129], r[130], r[131]);
    SplitProduct(ax, by, by, dy,  r[132], r[133], r[134], r[135]);
    SplitProduct(ax, ax, cx, dy,  r[136], r[137], r[138], r[139]);
    SplitProduct(ay, ay, cx, dy,  r[140], r[141], r[142], r[143]);
    SplitProduct(-bx, bx, cx, dy, r[144], r[145], r[146], r[147]);
    SplitProduct(-by, by, cx, dy, r[148], r[149], r[150], r[151]);
    SplitProduct(-ax, cx, cx, dy, r[152], r[153], r[154], r[155]);
    SplitProduct(bx, cx, cx, dy,  r[156], r[157], r[158], r[159]);

    SplitProduct(-ax, cy, cy, dy, r[160], r[161], r[162], r[163]);  
    SplitProduct(bx, cy, cy, dy,  r[164], r[165], r[166], r[167]);
    SplitProduct(ay, bx, dy, dy,  r[168], r[169], r[170], r[171]);
    SplitProduct(-ax, by, dy, dy, r[172], r[173], r[174], r[175]);
    SplitProduct(-ay, cx, dy, dy, r[176], r[177], r[178], r[179]);
    SplitProduct(by, cx, dy, dy,  r[180], r[181], r[182], r[183]);  
    SplitProduct(ax, cy, dy, dy,  r[184], r[185], r[186], r[187]);
    SplitProduct(-bx, cy, dy, dy, r[188], r[189], r[190], r[191]);
    
    real s = ExactSignSum<double, 192>(r);
    
    std::cout << std::endl
	      << "fp64 and incircleadapt give different answers: "
	      << "orient2d: " << ss << ", fp64: " << ddet
	      << ", exactSign: " << s << std::endl;
    //int qq; std::cin >> qq;
  }
#endif
  */
  
  return ss;
  
#else
  
  return incircleadapt(ax, ay, bx, by, cx, cy, dx, dy, permanent, pParam);
  
#endif
}

//######################################################################
// 
//######################################################################

//__host__ __device__
inline real Predicates::incircleadapt(real ax, real ay,
				      real bx, real by,
				      real cx, real cy,
				      real dx, real dy,
				      real permanent,
				      real *pParam) const
{
  real zero = 0.0;
  
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
  int axtbclen=0, aytbclen=0, bxtcalen=0, bytcalen=0, cxtablen=0, cytablen=0;
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
          + 2.0 * (adx * adxtail + ady * adytail) * (bdx * cdy - bdy * cdx))
       + ((bdx * bdx + bdy * bdy) * ((cdx * adytail + ady * cdxtail)
                                     - (cdy * adxtail + adx * cdytail))
          + 2.0 * (bdx * bdxtail + bdy * bdytail) * (cdx * ady - cdy * adx))
       + ((cdx * cdx + cdy * cdy) * ((adx * bdytail + bdy * adxtail)
                                     - (ady * bdxtail + bdx * adytail))
          + 2.0 * (cdx * cdxtail + cdy * cdytail) * (adx * bdy - ady * bdx));
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
    temp16alen = scale_expansion_zeroelim(axtbclen, axtbc, 2.0 * adx,
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
    temp16alen = scale_expansion_zeroelim(aytbclen, aytbc, 2.0 * ady,
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
    temp16alen = scale_expansion_zeroelim(bxtcalen, bxtca, 2.0 * bdx,
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
    temp16alen = scale_expansion_zeroelim(bytcalen, bytca, 2.0 * bdy,
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
    temp16alen = scale_expansion_zeroelim(cxtablen, cxtab, 2.0 * cdx,
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
    temp16alen = scale_expansion_zeroelim(cytablen, cytab, 2.0 * cdy,
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
      temp32alen = scale_expansion_zeroelim(axtbctlen, axtbct, 2.0 * adx,
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
      temp16alen = scale_expansion_zeroelim(axtbcttlen, axtbctt, 2.0 * adx,
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
      temp32alen = scale_expansion_zeroelim(aytbctlen, aytbct, 2.0 * ady,
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
      temp16alen = scale_expansion_zeroelim(aytbcttlen, aytbctt, 2.0 * ady,
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
      temp32alen = scale_expansion_zeroelim(bxtcatlen, bxtcat, 2.0 * bdx,
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
      temp16alen = scale_expansion_zeroelim(bxtcattlen, bxtcatt, 2.0 * bdx,
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
      temp32alen = scale_expansion_zeroelim(bytcatlen, bytcat, 2.0 * bdy,
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
      temp16alen = scale_expansion_zeroelim(bytcattlen, bytcatt, 2.0 * bdy,
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
      temp32alen = scale_expansion_zeroelim(cxtabtlen, cxtabt, 2.0 * cdx,
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
      temp16alen = scale_expansion_zeroelim(cxtabttlen, cxtabtt, 2.0 * cdx,
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
      temp32alen = scale_expansion_zeroelim(cytabtlen, cytabt, 2.0 * cdy,
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
      temp16alen = scale_expansion_zeroelim(cytabttlen, cytabtt, 2.0 * cdy,
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

#endif
}

#endif
