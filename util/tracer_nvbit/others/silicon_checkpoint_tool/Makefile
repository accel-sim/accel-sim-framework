SUB_DIRS        = $(wildcard */.)
SUB_DIRS_ALL    = $(SUB_DIRS:%=all-%)
SUB_DIRS_CLEAN  = $(SUB_DIRS:%=clean-%)

all: $(SUB_DIRS_ALL)
clean: $(SUB_DIRS_CLEAN)

$(SUB_DIRS_ALL):
	$(MAKE) $(MAKE_FLAGS) -C $(@:all-%=%)

$(SUB_DIRS_CLEAN):
	$(MAKE) $(MAKE_FLAGS) -C $(@:clean-%=%) clean
