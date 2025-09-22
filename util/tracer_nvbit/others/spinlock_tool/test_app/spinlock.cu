#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Function to reset the lock on device
__global__ void resetLock(volatile int* lock_ptr) {
    *lock_ptr = 0;
}

// Function to acquire the spinlock
__device__ void acquire_spinlock(volatile int* lock_ptr) {
    // Continuously try to acquire the lock using atomicCAS
    // It attempts to change the lock from 0 (unlocked) to 1 (locked)
    // The loop continues as long as the atomicCAS operation returns a non-zero value,
    // indicating that another thread held the lock and the swap failed.
    while (atomicCAS((int*)lock_ptr, 0, 1) != 0);
}

// Function to release the spinlock
__device__ void release_spinlock(volatile int* lock_ptr) {
    // Simply set the lock variable back to 0 (unlocked)
    *lock_ptr = 0;
    // Ensure the lock is visible to other threads
    __threadfence();
}

// Test kernel 1: Simple counter increment with spinlock
__global__ void testCounterKernel(int* data, int num_elements, int iterations, volatile int* lock_ptr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_elements) {
        for (int i = 0; i < iterations; i++) {
            // Acquire the lock before entering the critical section
            acquire_spinlock(lock_ptr);

            // Critical section: Increment shared counter
            data[0] += 1;

            // Release the lock after exiting the critical section
            release_spinlock(lock_ptr);
            
            // Ensure all threads reach this point before proceeding to next iter
            __syncthreads();
        }
    }
}

// Host function to check CUDA errors
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(1);
    }
}

// Host function to initialize and run tests
int main() {
    printf("Starting CUDA Spinlock Test Program\n");
    printf("===================================\n\n");

    // Allocate device memory for the lock
    volatile int* d_lock;
    checkCudaError(cudaMalloc(&d_lock, sizeof(int)), "Failed to allocate device lock memory");
    checkCudaError(cudaMemset((void*)d_lock, 0, sizeof(int)), "Failed to initialize device lock");

    // Test 1: Simple counter test
    printf("Simple spinlock test\n");
    printf("---------------------------\n");
    
    int* d_data;
    int h_data = 0;
    int num_threads = 1024;
    int iterations = 100;
    
    checkCudaError(cudaMalloc(&d_data, sizeof(int)), "Failed to allocate device memory");
    checkCudaError(cudaMemset(d_data, 0, sizeof(int)), "Failed to initialize device memory");
    
    // Launch kernel with multiple blocks and threads
    int block_size = 256;
    int grid_size = (num_threads + block_size - 1) / block_size;
    
    printf("Launching %d blocks with %d threads each (%d total threads)\n", 
           grid_size, block_size, num_threads);
    printf("Each thread will increment the counter %d times\n", iterations);

    testCounterKernel<<<grid_size, block_size>>>(d_data, num_threads, iterations, d_lock);
    checkCudaError(cudaGetLastError(), "Kernel execution failed");

    checkCudaError(cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost), 
                   "Failed to copy data from device");
    
    int expected = num_threads * iterations;
    printf("Expected result: %d\n", expected);
    printf("Actual result: %d\n", h_data);
    printf("Simple counter test %s\n\n", (h_data == expected) ? "PASSED" : "FAILED");

    // Cleanup
    cudaFree(d_data);
    cudaFree((void*)d_lock);
    
    return 0;
}