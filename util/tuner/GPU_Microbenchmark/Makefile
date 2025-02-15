
BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/bin
SUB_DIRS        = $(wildcard ubench/*/*/)
SUB_DIRS_ALL    = $(SUB_DIRS:%=all-%)
SUB_DIRS_CLEAN  = $(SUB_DIRS:%=clean-%)

all: create_dir $(SUB_DIRS_ALL)

clean: delete_dir $(SUB_DIRS_CLEAN)

$(SUB_DIRS_ALL):
	$(MAKE) $(MAKE_FLAGS) -C $(@:all-%=%)

$(SUB_DIRS_CLEAN):
	$(MAKE) $(MAKE_FLAGS) -C $(@:clean-%=%) clean

create_dir:
	mkdir -p $(BIN_DIR)

delete_dir:
	cd $(BIN_DIR); rm -f *
