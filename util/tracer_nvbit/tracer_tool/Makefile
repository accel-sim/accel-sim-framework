NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc)

ifeq ($(PTXAS_VER_CHECK), 0)
MAXRREGCOUNT_FLAG=-maxrregcount=24
else
MAXRREGCOUNT_FLAG=
endif

NVBIT_PATH=../nvbit_release/core
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=$(wildcard *.cu)

OBJECTS=$(SOURCES:.cu=.o)
ARCH?=all

ARCH_VER_REQ=11.7 # NVCC < 11.7 doesn't support -arch=all flag
ARCH_VER_CHECK=$(shell echo "${NVCC_VER} >= $(ARCH_VER_REQ)" | bc)
ifeq ($(ARCH_VER_CHECK),0)
ifeq ($(ARCH), all)
$(error ERROR: ARCH environment variable must be set to "sm_XX" for nvcc < 11.7, ex: run "export ARCH=sm_70 for compute capability 70")
endif
endif

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL=$(current_dir).so

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

%.o: %.cu common.h
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) -O3 -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu common.h
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.so *.o
