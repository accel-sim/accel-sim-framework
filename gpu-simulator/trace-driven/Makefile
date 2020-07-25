# Copyright (c) 2020 Timothy Rogers, Purdue University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# Neither the name of The University of British Columbia nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

DEBUG?=0


BUILD_DIR=../build/$(ACCELSIM_CONFIG)

ifeq ($(DEBUG),1)
	CXXFLAGS = -Wall -DDEBUG
else
	CXXFLAGS = -Wall
endif

ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS += -std=c++0x
endif

ifneq ($(DEBUG),1)
	OPTFLAGS += -O3
else
	CXXFLAGS += 
endif

OPTFLAGS += -g3 -fPIC -std=c++11

SRCS = $(shell ls *.cc)
EXCLUDES = 
CSRCS = $(filter-out $(EXCLUDES), $(SRCS))
OBJS = $(CSRCS:%.cc=$(BUILD_DIR)/%.o)

trace-driven:$(OBJS)

$(BUILD_DIR)/%.o: %.cc
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) -I$(CUDA_INSTALL_PATH)/include -I. -I$(GPGPUSIM_ROOT)/libcuda -I$(GPGPUSIM_ROOT)/src -c $*.cc -o $(BUILD_DIR)/$*.o 

$(BUILD_DIR)/trace-driven.Makefile.makedepend: depend

depend:
	touch $(BUILD_DIR)/trace-driven.Makefile.makedepend
	makedepend -f$(BUILD_DIR)/trace-driven.Makefile.makedepend -p$(BUILD_DIR)/ $(CSRCS) 2> /dev/null

include $(BUILD_DIR)/trace-driven.Makefile.makedepend
