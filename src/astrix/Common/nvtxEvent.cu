// -*-c++-*-
/*! \file nvtxEvent.cu
\brief Functions for nvtxEvent class

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "nvtxEvent.h"

namespace astrix {

uint32_t nvtxEvent::colors[] = {
  0x0000ff00,
  0x000000ff,
  0x00ffff00,
  0x00ff00ff,
  0x0000ffff,
  0x00ff0000,
  0x00ffffff
};

//#############################################################################
// Constructor
//#############################################################################

nvtxEvent::nvtxEvent(const char *name, int _colorID)
{
  colorID = _colorID % num_colors;

  // Set attributes
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = colors[colorID];
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = name;

  // Push event
  nvtxRangePushEx(&eventAttrib);
}

//#############################################################################
// Destructor
//#############################################################################

nvtxEvent::~nvtxEvent()
{
  // Pop event
  nvtxRangePop();
}

}
