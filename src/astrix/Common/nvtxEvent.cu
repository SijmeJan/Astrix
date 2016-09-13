// -*-c++-*-
/*! \file nvtxEvent.cu
\brief Functions for nvtxEvent class
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
