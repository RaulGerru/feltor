ifeq ($(strip $(HPC_SYSTEM)),m100)
#CC=xlc++ #C++ compiler
#MPICC=mpixlC  #mpi compiler
#CFLAGS=-Wall -std=c++1y -DWITHOUT_VCL -mcpu=power9 -qstrict# -mavx -mfma #flags for CC
#OMPFLAG=-qsmp=omp
CFLAGS=-Wall -std=c++14 -DWITHOUT_VCL -mcpu=power9 # -mavx -mfma #flags for CC
OPT=-O3 # optimization flags for host code
NVCC=nvcc #CUDA compiler
NVCCARCH=-arch sm_70 -Xcudafe "--diag_suppress=code_is_unreachable --diag_suppress=initialization_not_reachable" #nvcc gpu compute capability
NVCCFLAGS= -std=c++14 -Xcompiler "-mcpu=power9 -Wall"# -mavx -mfma" #flags for NVCC

INCLUDE += -I$(NETCDF_INC) -I$(HDF5_INC) -I$(JSONCPP_INC)
<<<<<<< HEAD
JSONLIB=-L$(JSONCPP_LIB) -ljsoncpp # json library for input parameters
=======
JSONLIB=-L$(JSONCPP_LIB) -ljsoncpp
#JSONLIB=-L$(HOME)/include/json/../../lib -ljsoncpp_static # json library for input parameters
>>>>>>> 91c98c0bd8b438c18ee2ea0052a8870939499f27
LIBS    =-L$(HDF5_LIB) -lhdf5 -lhdf5_hl
LIBS    +=-L$(NETCDF_LIB) -lnetcdf -lcurl
endif
#########################Modules to load ##################
#module load cuda
#module load gnu
#module load spectrum_mpi
#module load binutils/2.34
#
#module load zlib/1.2.11--gnu--8.4.0 
#module load szip/2.1.1--gnu--8.4.0
#module load hdf5/1.12.0--gnu--8.4.0
#module load netcdf/4.7.3--gnu--8.4.0
#module load jsoncpp/1.9.3--spectrum_mpi--10.3.1--binary
