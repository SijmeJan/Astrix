################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

MODULES := Mesh/Predicates Mesh/Coarsen Mesh/Param Mesh/Connectivity Mesh/Refine Mesh/Delaunay Mesh/Morton Mesh Array Simulation Common Device

BINDIR = ../../bin

# Create list of objects from .cu and .cpp in module directories
OBJ := $(patsubst %.cu,%.o,$(wildcard *.cu)) $(patsubst %.cpp,%.o,$(wildcard *.cpp)) $(foreach sdir,$(MODULES),$(patsubst %.cu,%.o,$(wildcard $(sdir)/*.cu))) $(foreach sdir,$(MODULES),$(patsubst %.cpp,%.o,$(wildcard $(sdir)/*.cpp)))

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

NVCC:= nvcc
CUDA_COMPUTE ?= 0

# internal flags
NVCCFLAGS   := -m${OS_SIZE}
CCFLAGS     := 
NVCCLDFLAGS :=
LDFLAGS     := 

# Extra user flags
EXTRA_NVCCFLAGS   ?= -O3 -lineinfo
#-g -G 
#--ptxas-options=-v
EXTRA_NVCCLDFLAGS ?=
EXTRA_LDFLAGS     ?= -lnvToolsExt
EXTRA_CCFLAGS     ?= -O3 -Wall -Wno-unused-private-field
#-Wno-unused-function -Wno-unused-const-variable 

# Compiler flags
ALL_CCFLAGS :=
# Add flags for nvcc compiler
ALL_CCFLAGS += $(NVCCFLAGS)
# Add flags for gcc compiler
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
# Add user flags for nvcc compiler
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
# Add user flags for gcc compiler
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

# Linker flags
ALL_LDFLAGS :=
# Add all flags for compiler
ALL_LDFLAGS += $(ALL_CCFLAGS)
# Add flags for nvcc linker
ALL_LDFLAGS += $(NVCCLDFLAGS)
# Add flags for gcc linker
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
# Add user flags for nvcc linker
ALL_LDFLAGS += $(EXTRA_NVCCLDFLAGS)
# Add user flags for gcc linker
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := 
LIBRARIES := 

##############################################################################
# CUDA code generation flags
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)
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
all: build

build: $(BINDIR)/astrix

$(BINDIR)/astrix: $(OBJ)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

##############################################################################
# Dependency rules
##############################################################################

main.o: main.cpp Common/definitions.h Simulation/simulation.h Device/device.h

##################
# Common sources #
##################

Common/cudaLow.o: Common/cudaLow.cpp Common/cudaLow.h
Common/nvtxEvent.o: Common/nvtxEvent.cu Common/nvtxEvent.h

##################
# Device sources #
##################

Device/device.o: Device/device.cpp Device/device.h

######################
# Simulation sources #
######################

Simulation/simulation.o: Simulation/simulation.cpp Common/definitions.h Simulation/simulation.h Mesh/mesh.h Array/array.h Device/device.h
Simulation/save.o: Simulation/save.cpp Common/definitions.h Simulation/simulation.h Mesh/mesh.h Array/array.h
Simulation/refine.o: Simulation/refine.cpp Common/definitions.h Simulation/simulation.h Mesh/mesh.h Array/array.h
Simulation/potential.o: Simulation/potential.cu Common/definitions.h Array/array.h Simulation/simulation.h Mesh/mesh.h Common/cudaLow.h
Simulation/spaceres.o: ALL_CCFLAGS += -maxrregcount=96
Simulation/spaceres.o: Simulation/spaceres.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h
Simulation/totalresNtot.o: ALL_CCFLAGS += -maxrregcount=96
Simulation/totalresNtot.o: Simulation/totalresNtot.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h
Simulation/totalresLDA.o: ALL_CCFLAGS += -maxrregcount=64
Simulation/totalresLDA.o: Simulation/totalresLDA.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h
Simulation/timestep.o: ALL_CCFLAGS += -maxrregcount=32
Simulation/timestep.o: Simulation/timestep.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/atomic.h Common/cudaLow.h
Simulation/timeloop.o: Simulation/timeloop.cpp Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/nvtxEvent.h
Simulation/initial.o: Simulation/initial.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h
Simulation/param.o: Simulation/param.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h
Simulation/energy.o: Simulation/energy.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h

Simulation/boundary_riemann.o: Simulation/boundary_riemann.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h
Simulation/boundary_nonreflect.o: Simulation/boundary_nonreflect.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h
Simulation/boundary_reflect.o: Simulation/boundary_reflect.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h
Simulation/boundary_extrapolate.o: Simulation/boundary_extrapolate.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h

Simulation/blend.o: ALL_CCFLAGS += -maxrregcount=32
Simulation/blend.o: Simulation/blend.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h
Simulation/blendx.o: ALL_CCFLAGS += -maxrregcount=32
Simulation/blendx.o: Simulation/blendx.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h
Simulation/minmaxvel.o: ALL_CCFLAGS += -maxrregcount=32
Simulation/minmaxvel.o: Simulation/minmaxvel.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h
Simulation/unphysical.o: Simulation/unphysical.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h
Simulation/replaceLDA.o: Simulation/replaceLDA.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h
Simulation/update.o: Simulation/update.cpp Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h 
Simulation/update_addresidue.o: ALL_CCFLAGS += -maxrregcount=48
Simulation/update_addresidue.o: Simulation/update_addresidue.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/atomic.h
Simulation/mass.o: Simulation/mass.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h
Simulation/kh.o: Simulation/kh.cu Common/definitions.h Mesh/mesh.h Simulation/simulation.h Array/array.h Common/cudaLow.h Common/inlineMath.h

################
# Mesh sources #
################

Mesh/mesh.o: Mesh/mesh.cpp Mesh/Predicates/predicates.h Common/definitions.h Mesh/mesh.h Array/array.h Mesh/Delaunay/delaunay.h Mesh/Refine/refine.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/edgenormal.o: Mesh/edgenormal.cu Common/definitions.h Mesh/mesh.h Array/array.h Common/cudaLow.h Common/inlineMath.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/vertexarea.o: Mesh/vertexarea.cu Common/definitions.h Mesh/mesh.h Array/array.h Common/atomic.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/error.o: Mesh/error.cu Common/definitions.h Array/array.h Mesh/mesh.h Common/atomic.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/save.o: Mesh/save.cpp Common/definitions.h Mesh/mesh.h Array/array.h Mesh/Connectivity/connectivity.h
Mesh/debug.o: Mesh/debug.cu Common/definitions.h Mesh/mesh.h Array/array.h Mesh/Predicates/predicates.h Common/inlineMath.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/boundary.o: Mesh/boundary.cu Array/array.h Common/definitions.h Mesh/mesh.h Common/cudaLow.h Mesh/Delaunay/delaunay.h Mesh/Refine/refine.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/findboundaryvertices.o: Mesh/findboundaryvertices.cu Common/definitions.h Mesh/mesh.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/structured.o: Mesh/structured.cpp Common/definitions.h Mesh/mesh.h Array/array.h Mesh/Morton/morton.h Mesh/Delaunay/delaunay.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h

Mesh/improvequality.o: Mesh/improvequality.cpp Array/array.h Common/definitions.h Mesh/mesh.h Mesh/Morton/morton.h Mesh/Refine/refine.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/wantrefine.o: Mesh/wantrefine.cu Array/array.h Common/definitions.h Mesh/mesh.h Common/cudaLow.h Mesh/Param/meshparameter.h
Mesh/coarsen.o: Mesh/coarsen.cpp Common/definitions.h Array/array.h Mesh/mesh.h Common/nvtxEvent.h Mesh/Delaunay/delaunay.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h Mesh/Coarsen/coarsen.h

###########################
# Mesh/Predicates sources #
###########################

Mesh/Predicates/predicates.o: Mesh/Predicates/predicates.cu Mesh/Predicates/predicates.h Common/definitions.h Array/array.h Device/device.h

#########################
# Mesh/Coarsen sources #
#########################

Mesh/Coarsen/coarsen.o: Mesh/Coarsen/coarsen.cpp Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h
Mesh/Coarsen/removevertices.o: Mesh/Coarsen/removevertices.cpp Common/definitions.h Array/array.h Mesh/Coarsen/coarsen.h Common/nvtxEvent.h Mesh/Delaunay/delaunay.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h Mesh/Predicates/predicates.h
Mesh/Coarsen/encroach.o: Mesh/Coarsen/encroach.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Mesh/Predicates/predicates.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Coarsen/remove.o: Mesh/Coarsen/remove.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h
Mesh/Coarsen/adjuststate.o: Mesh/Coarsen/adjuststate.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Coarsen/vertextriangle.o: Mesh/Coarsen/vertextriangle.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/cudaLow.h Common/atomic.h Mesh/Connectivity/connectivity.h
Mesh/Coarsen/maxtripervert.o: Mesh/Coarsen/maxtripervert.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/cudaLow.h Common/atomic.h Mesh/Connectivity/connectivity.h
Mesh/Coarsen/flagvertex.o: Mesh/Coarsen/flagvertex.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/cudaLow.h Common/atomic.h Mesh/Connectivity/connectivity.h
Mesh/Coarsen/trianglelist.o: Mesh/Coarsen/trianglelist.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Coarsen/allowedtarget.o: Mesh/Coarsen/allowedtarget.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/cudaLow.h Mesh/Predicates/predicates.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Coarsen/target.o: Mesh/Coarsen/target.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Coarsen/neighbour.o: Mesh/Coarsen/neighbour.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Coarsen/reject.o: Mesh/Coarsen/reject.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Coarsen/findparalleldeletion.o: Mesh/Coarsen/findparalleldeletion.cu Array/array.h Common/definitions.h Mesh/Coarsen/coarsen.h Mesh/Coarsen/unique_triangle.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Coarsen/vertexarea.o: Mesh/Coarsen/vertexarea.cu Common/definitions.h Mesh/Coarsen/coarsen.h Array/array.h Common/atomic.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Coarsen/unique_triangle.o: Mesh/Coarsen/unique_triangle.cu Mesh/Coarsen/unique_triangle.h Array/array.h Common/cudaLow.h

#########################
# Mesh/Refine sources #
#########################

Mesh/Refine/addtoperiodic.o: Mesh/Refine/addtoperiodic.cu Common/definitions.h Mesh/Refine/refine.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Refine/addvertex.o: Mesh/Refine/addvertex.cpp Array/array.h Common/definitions.h Mesh/Refine/refine.h Common/nvtxEvent.h Mesh/Delaunay/delaunay.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Refine/badtriangle.o: Mesh/Refine/badtriangle.cu Array/array.h Common/definitions.h Mesh/Refine/refine.h Common/cudaLow.h Common/nvtxEvent.h Common/inlineMath.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Refine/circum.o: Mesh/Refine/circum.cu Array/array.h Common/definitions.h Mesh/Refine/refine.h Common/cudaLow.h Common/nvtxEvent.h Common/inlineMath.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Refine/findparallelinsertion.o: Mesh/Refine/findparallelinsertion.cu Array/array.h Common/definitions.h Mesh/Refine/refine.h Common/cudaLow.h Common/nvtxEvent.h Mesh/Connectivity/connectivity.h
Mesh/Refine/findtriangles.o: Mesh/Refine/findtriangles.cu Common/definitions.h Mesh/Refine/refine.h Array/array.h Mesh/Predicates/predicates.h Common/cudaLow.h Common/nvtxEvent.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Refine/flagsegment.o: Mesh/Refine/flagsegment.cu Common/definitions.h Mesh/Refine/refine.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h 
Mesh/Refine/improve.o: Mesh/Refine/improve.cpp Array/array.h Common/definitions.h Mesh/Refine/refine.h Common/nvtxEvent.h Mesh/Morton/morton.h Mesh/Delaunay/delaunay.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Refine/insert.o: Mesh/Refine/insert.cu Common/definitions.h Mesh/Refine/refine.h Array/array.h Mesh/Predicates/predicates.h Common/cudaLow.h Common/nvtxEvent.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Refine/interpolatestate.o: Mesh/Refine/interpolatestate.cu Common/definitions.h Mesh/Refine/refine.h Array/array.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Refine/refine.o: Mesh/Refine/refine.cpp Array/array.h Common/definitions.h Mesh/Refine/refine.h Common/nvtxEvent.h Mesh/Morton/morton.h Mesh/Delaunay/delaunay.h
Mesh/Refine/splitsegment.o: Mesh/Refine/splitsegment.cpp Common/definitions.h Mesh/Refine/refine.h Array/array.h Common/cudaLow.h Common/nvtxEvent.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Refine/testencroach.o: ALL_CCFLAGS += -maxrregcount=52
Mesh/Refine/testencroach.o: Mesh/Refine/testencroach.cu Common/definitions.h Mesh/Refine/refine.h Array/array.h Common/cudaLow.h Common/nvtxEvent.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Refine/testrefine.o: Mesh/Refine/testrefine.cu Array/array.h Common/definitions.h Mesh/Refine/refine.h Common/cudaLow.h Common/nvtxEvent.h Common/inlineMath.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h

#########################
# Mesh/Delaunay sources #
#########################

Mesh/Delaunay/adjuststate.o: Mesh/Delaunay/adjuststate.cu Mesh/Predicates/predicates.h Common/definitions.h Mesh/Delaunay/delaunay.h Array/array.h Common/cudaLow.h Common/atomic.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Delaunay/checkedge.o: ALL_CCFLAGS += -maxrregcount=32
Mesh/Delaunay/checkedge.o: Mesh/Delaunay/checkedge.cu Mesh/Predicates/predicates.h Common/definitions.h Mesh/Delaunay/delaunay.h Array/array.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Delaunay/checkflop.o: ALL_CCFLAGS += -maxrregcount=32
Mesh/Delaunay/checkflop.o: Mesh/Delaunay/checkflop.cu Mesh/Predicates/predicates.h Common/definitions.h Mesh/Delaunay/delaunay.h Array/array.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Delaunay/delaunay.o: Mesh/Delaunay/delaunay.cpp Common/definitions.h Mesh/Delaunay/delaunay.h Array/array.h
Mesh/Delaunay/edgerepair.o: Mesh/Delaunay/edgerepair.cu Common/definitions.h Mesh/Delaunay/delaunay.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Delaunay/fillsub.o: Mesh/Delaunay/fillsub.cu Common/definitions.h Mesh/Delaunay/delaunay.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Delaunay/findparallelflip.o: Mesh/Delaunay/findparallelflip.cu Common/definitions.h Mesh/Delaunay/delaunay.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Delaunay/flipedge.o: Mesh/Delaunay/flipedge.cu Common/definitions.h Mesh/Delaunay/delaunay.h Array/array.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h
Mesh/Delaunay/makedelaunay.o: Mesh/Delaunay/makedelaunay.cpp Common/definitions.h Mesh/Delaunay/delaunay.h Array/array.h Common/nvtxEvent.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h
Mesh/Delaunay/vertexarea.o: Mesh/Delaunay/vertexarea.cu Common/definitions.h Mesh/Delaunay/delaunay.h Array/array.h Common/atomic.h Common/cudaLow.h Mesh/triangleLow.h Mesh/Connectivity/connectivity.h Mesh/Param/meshparameter.h

#######################
# Mesh/Morton sources #
#######################

Mesh/Morton/edge.o: Mesh/Morton/edge.cu Common/definitions.h Mesh/Morton/morton.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Morton/morton.o: Mesh/Morton/morton.cpp Common/definitions.h Mesh/Morton/morton.h Array/array.h
Mesh/Morton/order.o: Mesh/Morton/order.cpp Common/definitions.h Mesh/Morton/morton.h Array/array.h Common/nvtxEvent.h Mesh/Connectivity/connectivity.h
Mesh/Morton/triangle.o: Mesh/Morton/triangle.cu Common/definitions.h Mesh/Morton/morton.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Morton/values.o: Mesh/Morton/values.cu Common/definitions.h Mesh/Morton/morton.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h
Mesh/Morton/vertex.o: Mesh/Morton/vertex.cpp Common/definitions.h Mesh/Morton/morton.h Array/array.h Common/cudaLow.h Mesh/Connectivity/connectivity.h

#############################
# Mesh/Connectivity sources #
#############################

Mesh/Connectivity/connectivity.o: Mesh/Connectivity/connectivity.cpp Common/definitions.h Array/array.h Mesh/Connectivity/connectivity.h

#############################
# Mesh/Param sources #
#############################

Mesh/Param/checkvalidity.o: Mesh/Param/checkvalidity.cpp Common/definitions.h Mesh/Param/meshparameter.h
Mesh/Param/meshparameter.o: Mesh/Param/meshparameter.cpp Common/definitions.h Mesh/Param/meshparameter.h
Mesh/Param/readfromfile.o: Mesh/Param/readfromfile.cpp Common/definitions.h Mesh/Param/meshparameter.h

#######################
# Array sources       #
#######################

Array/array.o: Array/array.cu Array/array.h Common/cudaLow.h
Array/sort.o: Array/sort.cu Array/array.h Common/cudaLow.h
Array/gather.o: Array/gather.cu Array/array.h Common/cudaLow.h
Array/random.o: Array/random.cu Array/array.h Common/cudaLow.h
Array/size.o: Array/size.cu Array/array.h Common/cudaLow.h
Array/equal.o: Array/equal.cu Array/array.h Common/cudaLow.h
Array/transform.o: Array/transform.cu Array/array.h Common/cudaLow.h
Array/single.o: Array/single.cu Array/array.h Common/cudaLow.h
Array/invert.o: Array/invert.cu Array/array.h Common/cudaLow.h
Array/value.o: Array/value.cu Array/array.h Common/cudaLow.h
Array/series.o: Array/series.cu Array/array.h Common/cudaLow.h
Array/reindex.o: Array/reindex.cu Array/array.h Common/cudaLow.h
Array/compact.o: Array/compact.cu Array/array.h Common/cudaLow.h
Array/scan.o: Array/scan.cu Array/array.h Common/cudaLow.h
Array/reduce.o: Array/reduce.cu Array/array.h Common/cudaLow.h
Array/add.o: Array/add.cu Array/array.h Common/cudaLow.h
Array/diff.o: Array/diff.cu Array/array.h Common/cudaLow.h
Array/removeevery.o: Array/removeevery.cu Array/array.h Common/cudaLow.h
Array/select.o: Array/select.cu Array/array.h Common/cudaLow.h
Array/unique.o: Array/unique.cu Array/array.h Common/cudaLow.h

################################################################################
# Rules to convert .cpp and .cu into .o
################################################################################

%.o:%.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -dc -o $@ -c $<
%.o:%.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -dc -o $@ -c $<

################################################################################
# Clean up 
################################################################################

clean:
	$(foreach sdir,$(MODULES),rm -f $(sdir)/*.o $(sdir)/*~ $(sdir)/*.ii $(sdir)/*.i $(sdir)/*.cubin $(sdir)/*.cu.cpp $(sdir)/*.cudafe* $(sdir)/*.fatbin* $(sdir)/*.hash $(sdir)/*.ptx $(sdir)/*.module*)
	rm -f *.o *~ *.ii *.i *.cubin *.cu.cpp *.cudafe* *.fatbin* *.hash *.ptx *.module*
	rm -f $(BINDIR)/astrix
