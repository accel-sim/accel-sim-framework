spinlock: spinlock.cu
	nvcc -o spinlock -arch=sm_70 spinlock.cu

sass:
	cuobjdump -sass spinlock > spinlock.sass

clean:
	rm -f spinlock
	rm -f spinlock.sass