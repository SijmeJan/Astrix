// -*-c++-*-
/*! \file mersenne.cpp
\brief Functions for pseudo random number generation

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <iostream>
#include <stdexcept>
#include <cmath>

#include "../array.h"
#include "./mersenne.h"

namespace astrix {

//#########################################################################
/*! Create Mersenne twister pseudo random number generator.

\param seed Seed value to initialize generator*/
//#########################################################################

Mersenne::Mersenne(uint32_t seed)
{
  MT[0] = seed;
  index = SIZE;

  for (uint_fast32_t i = 1; i < SIZE; ++i)
    MT[i] = 0x6c078965*(MT[i - 1] ^ MT[i - 1] >> 30) + i;

}

//#########################################################################
//#########################################################################

Mersenne::~Mersenne()
{
}


//#########################################################################
//#########################################################################

uint32_t Mersenne::rand_u32()
{
  if ( index == SIZE ) {
    generate_numbers();
    index = 0;
  }

  return MT_TEMPERED[index++];
}


void Mersenne::generate_numbers()
{
  /*
   * For performance reasons, we've unrolled the loop three times, thus
   * mitigating the need for any modulus operations. Anyway, it seems this
   * trick is old hat: http://www.quadibloc.com/crypto/co4814.htm
   */

  size_t i = 0;
  uint32_t y;

  // i = [0 ... 226]
  while ( i < DIFF ) {
    y = M32(MT[i]) | L31(MT[i+1]);
    MT[i] = MT[i + PERIOD] ^ (y >> 1) ^ (((int32_t(y) << 31) >> 31) & MAGIC);
    ++i;
  }

  // i = [227 ... 622]
  while ( i < SIZE -1 ) {
    /*
     * 623-227 = 396 = 2*2*3*3*11, so we can unroll this loop in any number
     * that evenly divides 396 (2, 4, 6, etc). Here we'll unroll 11 times.
     */
    y = M32(MT[i]) | L31(MT[i+1]);
    MT[i] = MT[i - DIFF] ^ (y >> 1) ^ (((int32_t(y) << 31) >> 31) & MAGIC);
    ++i;
  }

  // i = 623, last step rolls over
  y = M32(MT[SIZE-1]) | L31(MT[0]);
  MT[SIZE-1] = MT[PERIOD-1] ^ (y >> 1) ^ (((int32_t(y) << 31) >> 31) & MAGIC);

  // Temper all numbers in a batch
  for (size_t i = 0; i < SIZE; ++i) {
    y = MT[i];
    y ^= y >> 11;
    y ^= y << 7  & 0x9d2c5680;
    y ^= y << 15 & 0xefc60000;
    y ^= y >> 18;
    MT_TEMPERED[i] = y;
  }

  index = 0;
}

}  // namespace astrix
