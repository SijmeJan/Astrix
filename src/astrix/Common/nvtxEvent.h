/*! \file nvtxEvent.h
\brief Header file containing nvtxEvent class definition

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ASTRIX_NVTX_EVENT_H
#define ASTRIX_NVTX_EVENT_H

#include <nvToolsExt.h>

namespace astrix {

//! Class handling NVTX events for use with the NVIDIA profiler
/*! The NVIDIA Visual Profiler allows for user-defined colors to appear in the time line to easily identify functions that take up most of the time. Creating an nvtxEvent object starts such a colored time line, while destroying it ends the time line.*/
class nvtxEvent
{
 public:
  //! Create NVTX event
  /*! Create NVTX event with name and color.
    \param *name Name (will appear in profiler)
    \param _colorID Color ID (range 0-6)
  */
  nvtxEvent(const char *name, int _colorID);
  //! Destroy NVTX event
  ~nvtxEvent();

 private:
  //! Color for this event
  int colorID;
  //! Number of available colors
  static const int num_colors = 7;
  //! Available colors (see nvtxEvent.cu)
  static uint32_t colors[num_colors];
};

}

#endif
