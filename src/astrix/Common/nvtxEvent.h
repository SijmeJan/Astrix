/*! \file nvtxEvent.h
\brief Header file containing nvtxEvent class definition
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
