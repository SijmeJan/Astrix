################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms
#
# Requires that either nvcc can be found in the PATH or else that 
# CUDA_INSTALL_PATH contains the CUDA installation directory (e.g.
# /usr/local/cuda). By default, compile for all possible target architectures
# (slow). Use for example CUDA_COMPUTE=30 to compile for a 3.0 compute
# capability only. Use CUDA_PROFILE=1 to compile for profiling, and
# CUDA_DEBUG=1 to compile for debugging. 
#
################################################################################

# Guess CUDA install path from nvcc location
CUDA_INSTALL_PATH ?= $(shell which nvcc | sed "s/\/bin\/nvcc//")

# If not specified, compile for all supported compute capabilities 
CUDA_COMPUTE ?= 0

# By default, no profiling flags
CUDA_PROFILE ?= 0

# By default, no debug flags
CUDA_DEBUG ?= 0

# Directory to put binaries in
BINDIR = ../../bin

# In case SHELL is inherited from OS
SHELL = /bin/sh
# No implicit rules needed
.SUFFIXES:

################################################################################
# Create list of source, object and dependency files
################################################################################

# List of modules (must be directories in src/astrix)
MODULES := Mesh/Predicates Mesh/Coarsen Mesh/Param Mesh/Connectivity Mesh/Refine Mesh/Delaunay Mesh/Morton Mesh Array Simulation Common Device

# Create list of source files in module directories: list all .cpp and .cu files
SRC :=  $(wildcard *.cu) $(wildcard *.cpp) $(foreach sdir,$(MODULES),$(wildcard $(sdir)/*.cu)) $(foreach sdir,$(MODULES),$(wildcard $(sdir)/*.cpp))

# Create list of objects from .cu and .cpp in module directories
OBJ = $(patsubst %.cu,%.o,$(patsubst %.cpp,%.o,$(SRC)))

# Create list of dependency files from .cu and .cpp in module directories
DEP = $(patsubst %.cu,%.d,$(patsubst %.cpp,%.d,$(SRC)))

################################################################################
# Compiler and linker flags
################################################################################

# NVCC compiler 
NVCC:= $(CUDA_INSTALL_PATH)/bin/nvcc

# Detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")

# If on 64 bit system, use lib64/ rather than lib/ if lib64/ exists
LIB_32 := $(CUDA_INSTALL_PATH)/lib
LIB_64 := $(CUDA_INSTALL_PATH)/lib64
WTEMP:=$(wildcard $(LIB_64))

LIB_PATH := $(LIB_32)
ifeq ($(OS_SIZE), 64)
	ifeq ($(LIB_64),$(WTEMP))
		LIB_PATH := $(LIB_64)
	endif
endif

# Standard compiler and linker flags
NVCCFLAGS   := -m${OS_SIZE} -O3
CCFLAGS     := -Wall -Wno-unused-private-field
NVCCLDFLAGS :=
LDFLAGS     := -lnvToolsExt -rpath $(LIB_PATH)

# Add lineinfo and register use info if profiling
ifeq ($(CUDA_PROFILE),1)
	NVCCFLAGS += -lineinfo --ptxas-options=-v -DTIME_ASTRIX
endif
# Add debug info if debugging
ifeq ($(CUDA_DEBUG),1)
	NVCCFLAGS += -g -G 
endif

# Compiler flags
ALL_CCFLAGS :=
# Add flags for nvcc compiler
ALL_CCFLAGS += $(NVCCFLAGS)
# Add flags for gcc compiler
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

# Linker flags
ALL_LDFLAGS :=
# Add all flags for compiler
ALL_LDFLAGS += $(ALL_CCFLAGS)
# Add flags for nvcc linker
ALL_LDFLAGS += $(NVCCLDFLAGS)
# Add flags for gcc linker
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I$(CUDA_INSTALL_PATH)/include
LIBRARIES := -L$(LIB_PATH)

################################################################################
# CUDA code generation flags
################################################################################

# Virtual architecture, as low as possible (compute_20 is deprecated)
CUDA_ARCH := compute_30

# By default, generate code for all architectures
CUDA_SM_ARCH := sm_30,sm_32,sm_35,sm_50,sm_52,sm_53

# If CUDA_COMPUTE is set, specialize appropriately
ifeq ($(CUDA_COMPUTE),30)
	CUDA_SM_ARCH := sm_30
endif
ifeq ($(CUDA_COMPUTE),32)
	CUDA_SM_ARCH := sm_32
endif
ifeq ($(CUDA_COMPUTE),35)
	CUDA_SM_ARCH := sm_35
endif
ifeq ($(CUDA_COMPUTE),50)
	CUDA_SM_ARCH := sm_50
endif
ifeq ($(CUDA_COMPUTE),52)
	CUDA_SM_ARCH := sm_52
endif
ifeq ($(CUDA_COMPUTE),53)
	CUDA_SM_ARCH := sm_53
endif

GENCODE_FLAGS := -arch=$(CUDA_ARCH) -code=$(CUDA_SM_ARCH)

################################################################################
# Target rules
################################################################################

all: $(BINDIR)/astrix

# Build Astrix executable
$(BINDIR)/astrix: $(OBJ)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

# Clean up
clean:
	$(foreach sdir,$(MODULES),rm -f $(sdir)/*.o $(sdir)/*.d $(sdir)/*~ $(sdir)/*.ii $(sdir)/*.i $(sdir)/*.cubin $(sdir)/*.cu.cpp $(sdir)/*.cudafe* $(sdir)/*.fatbin* $(sdir)/*.hash $(sdir)/*.ptx $(sdir)/*.module*)
	rm -f *.o *.d *~ *.ii *.i *.cubin *.cu.cpp *.cudafe* *.fatbin* *.hash *.ptx *.module*
	rm -f $(BINDIR)/astrix
	-rm -f -r $(BINDIR)/astrix.dSYM

################################################################################
# Rules to convert .cpp and .cu into .o. Generate new dependency file .d
################################################################################

%.o:%.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -dc -o $@ -c $<
	$(NVCC) -Wno-deprecated-gpu-targets -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MT $@ -MM" -o $*.d $<
%.o:%.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -dc -o $@ -c $<
	$(NVCC) -Wno-deprecated-gpu-targets -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MT $@ -MM" -o $*.d $<

# Keep .d files
.PRECIOUS: %.d

##############################################################################
# Include dependency files
##############################################################################

-include $(DEP)

##############################################################################
# Register limits
##############################################################################

Simulation/spaceres.o: ALL_CCFLAGS += -maxrregcount=96
Simulation/totalresNtot.o: ALL_CCFLAGS += -maxrregcount=96
Simulation/totalresLDA.o: ALL_CCFLAGS += -maxrregcount=64
Simulation/timestep.o: ALL_CCFLAGS += -maxrregcount=32
Simulation/blend.o: ALL_CCFLAGS += -maxrregcount=32
Simulation/blendx.o: ALL_CCFLAGS += -maxrregcount=32
Simulation/minmaxvel.o: ALL_CCFLAGS += -maxrregcount=32
Simulation/update_addresidue.o: ALL_CCFLAGS += -maxrregcount=48

Mesh/Refine/testencroach.o: ALL_CCFLAGS += -maxrregcount=52
Mesh/Refine/lock.o: ALL_CCFLAGS += -maxrregcount=52
Mesh/Delaunay/checkedge.o: ALL_CCFLAGS += -maxrregcount=32
Mesh/Delaunay/checkflop.o: ALL_CCFLAGS += -maxrregcount=32
