################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
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

# Standard compiler and linker flags
NVCCFLAGS   := -m${OS_SIZE} -O3
CCFLAGS     := -Wall -Wno-unused-private-field
NVCCLDFLAGS :=
LDFLAGS     := -lnvToolsExt

# Add lineinfo and register use info if profiling
ifeq ($(CUDA_PROFILE),1)
	NVCCFLAGS += -lineinfo --ptxas-options=-v
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
LIBRARIES := -L$(CUDA_INSTALL_PATH)/lib
ifeq ($(OS_SIZE), 64)
	LIBRARIES := -L$(CUDA_INSTALL_PATH)/lib64
endif

################################################################################
# CUDA code generation flags
################################################################################

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=\"sm_35,compute_35\"

# By default, generate all
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)
# If CUDA_COMPUTE is set, generate code for specific architecture
ifeq ($(CUDA_COMPUTE),20)
	GENCODE_FLAGS := $(GENCODE_SM20)
endif
ifeq ($(CUDA_COMPUTE),30)
	GENCODE_FLAGS := $(GENCODE_SM30)
endif
ifeq ($(CUDA_COMPUTE),35)
	GENCODE_FLAGS := $(GENCODE_SM35)
endif

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
	$(NVCC) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MT $@ -MM" -o $*.d $<
%.o:%.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -dc -o $@ -c $<
	$(NVCC) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MT $@ -MM" -o $*.d $<

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

Mesh/Delaunay/checkedge.o: ALL_CCFLAGS += -maxrregcount=32
Mesh/Delaunay/checkflop.o: ALL_CCFLAGS += -maxrregcount=32
