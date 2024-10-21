import sys
import os
    
import glob

# Import renderdoc if not already imported (e.g. in the UI)
if 'renderdoc' not in sys.modules and '_renderdoc' not in sys.modules:
	import renderdoc

# Alias renderdoc for legibility
rd = renderdoc
all = {}

actions = {}
draws = 0
frame = 0

# Define a recursive function for iterating over actions
def iterDraw(d, indent = ''):
	global actions

	# save the action by eventId
	actions[d.eventId] = d

	# Iterate over the draw's children
	for d in d.children:
		iterDraw(d, indent + '    ')

def sampleCode(controller):
	# Iterate over all of the root actions, so we have names for each
	# eventId
	for d in controller.GetRootActions():
		iterDraw(d)

	# Enumerate the available counters
	counters = controller.EnumerateCounters()

	if not (rd.GPUCounter.SamplesPassed in counters):
		raise RuntimeError("Implementation doesn't support Samples Passed counter")

	# Now we fetch the counter data, this is a good time to batch requests of as many
	# counters as possible, the implementation handles any book keeping.
	results = controller.FetchCounters([rd.GPUCounter.SamplesPassed])

	# Get the description for the counter we want
	samplesPassedDesc = controller.DescribeCounter(3000247)

	# Describe each counter
	# for c in counters:
	# 	desc = controller.DescribeCounter(c)

	# 	print("Counter %d (%s):" % (c, desc.name))
	# 	print("    %s" % desc.description)
	# 	print("    Returns %d byte %s, representing %s" % (desc.resultByteWidth, desc.resultType, desc.unit))

	# Look in the results for any draws with 0 samples written - this is an indication
	# that if a lot of draws appear then culling could be better.
	drawtime = []
	for r in results:
		draw = actions[r.eventId]

		# Only care about draws, not about clears and other misc events
		if not (draw.flags & rd.ActionFlags.Drawcall):
			continue

		if samplesPassedDesc.resultByteWidth == 4:
			val = r.value.u32
		else:
			val = r.value.u64
		drawtime.append(val)
		print(r.eventId, val)
	global frame
	all[frame] = drawtime
	frame += 1
	# print(len(drawtime))

		

	# 	if r.eventId not in all.keys():
	# 		all[r.eventId] = []
	# 		for i in range(draws-1):
	# 			all[r.eventId].append(0)
	# 	all[r.eventId].append(val)
	# for (k,v) in all.items():
	# 	if len(v) < draws:
	# 		all[k].append(0)
	# 	assert(len(v) == draws)

def loadCapture(filename):
	# Open a capture file handle
	cap = rd.OpenCaptureFile()

	# Open a particular file - see also OpenBuffer to load from memory
	result = cap.OpenFile(filename, '', None)

	# Make sure the file opened successfully	# for c in counters:
	# 	desc = controller.DescribeCounter(c)

	# 	print("Counter %d (%s):" % (c, desc.name))
	# 	print("    %s" % desc.description)
	# 	print("    Returns %d byte %s, representing %s" % (desc.resultByteWidth, desc.resultType, desc.unit))
	if result != rd.ResultCode.Succeeded:
		raise RuntimeError("Couldn't open file: " + str(result))

	# Make sure we can replay
	if not cap.LocalReplaySupport():
		raise RuntimeError("Capture cannot be replayed")

	# Initialise the replay
	result,controller = cap.OpenCapture(rd.ReplayOptions(), None)

	if result != rd.ResultCode.Succeeded:
		raise RuntimeError("Couldn't initialise replay: " + str(result))

	return cap,controller

rd.InitialiseReplay(rd.GlobalEnvironment(), [])

# if len(sys.argv) <= 1:
#     print('Usage: python3 {} filename.rdc'.format(sys.argv[0]))
#     sys.exit(0)

dir_name = '/home/pan/Documents/sponza-capture/'
# Get list of all files in a given directory sorted by name
files = sorted( filter( os.path.isfile,
                        glob.glob(dir_name + '*') ) )
for file in files:
    path = "/home/pan/Documents/sponza-capture/" + file
    print(file)
    cap,controller = loadCapture(file)
    draws += 1

    sampleCode(controller)

    controller.Shutdown()
    cap.Shutdown()


for(k,v) in all.items():
	to_print = str(k) + ","
	for val in v:
		to_print += str(val) + ","
	print(to_print)
# for k in sorted(all.keys()):
# 	to_print = str(k) + ","
# 	for val in all[k]:
# 		to_print += str(val) + ","
# 	print(to_print)
		
rd.ShutdownReplay()