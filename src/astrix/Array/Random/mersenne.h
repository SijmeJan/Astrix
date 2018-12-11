/*! \file mersenne.h
\brief Header file for Mersenne twister class (host only!)

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#ifndef ASTRIX_MERSENNE_H
#define ASTRIX_MERSENNE_H

namespace astrix {

//! Class containing pseudo-random number generator
class Mersenne
{
 public:
  //! Constructor
  Mersenne(uint32_t seed);
  //! Destructor; releases memory.
  ~Mersenne();

  uint32_t rand_u32();

 private:
  static const size_t SIZE   = 624;
  static const size_t PERIOD = 397;
  static const size_t DIFF = SIZE - PERIOD;

  static const uint32_t MAGIC = 0x9908b0df;

  uint32_t MT[SIZE];
  uint32_t MT_TEMPERED[SIZE];
  size_t index;

  inline uint32_t M32(uint32_t x) { return 0x80000000 & x;}
  inline uint32_t L31(uint32_t x) { return 0x7FFFFFFF & x;}

  void generate_numbers();
};

} // namespace astrix

#endif  // ASTRIX_MERSENNE_H
