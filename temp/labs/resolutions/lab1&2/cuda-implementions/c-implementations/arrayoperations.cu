/**
 *
 * High Performance Computing
 * Practical Lab #1
 * Exercise 1)
 *
 * Description/Steps of operations performed by the GPU CUDA's Kernels:
 * - 1st) The sum of the content of two Arrays of Integers,
 *        resulting on a third Array of Integers,
 *        using CUDA-based parallel operations in the GPU.
 *
 * - 2nd) The decrement of the content of the third Array of Integers,
 *        using CUDA-based parallel operations in the GPU.
 *
 * - 3rd) The reverse of the content of the third Array of Integers,
 *        resulting on a fourth Array of Integers,
 *        using CUDA-based parallel operations in the GPU.
 *
 *   Implementation in C.
 *
 * Authors:
 * - Herve Miguel Paulino (Professor)
 * - Ruben Andre Barreiro
 *
 */

#include "../../../../include/cadlabs.hpp"
#include "../../../../include/timer.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <stdio.h>

#define THREADS_PER_BLOCK (512)
#define BLOCK_DIMENSIONS (1024 * 1024)


using namespace std;

/**
 *
 * Simple utility function to check for runtime errors in GPU CUDA's Kernels.
 *
 * @param msg the message related to the runtime errors in GPU CUDA's Kernels
 *
 */
void checkCUDAError(const char* msg);

/**
 *
 * 1st step of operations performed by the GPU CUDA's Kernels:
 *
 * The GPU CUDA's Kernel to perform a sum of all the two Integers contained in two Arrays,
 * performing the computing in parallel.
 *
 * @param a a pointer to the 1st Array of Integers
 * @param b a pointer to the 2nd Array of Integers
 * @param c a pointer to the 3rd Array of Integers
 *          for the final result of the operation
 * @param size_array the total size of the Arrays of Integers, previously allocated
 *
 */
__global__ void addKernel(const int* a, const int* b, int* c, const unsigned int size_array) {

    // Index of Kernel/Thread = (Number of Blocks x Size of Block) + Offset associated to the Thread,
    //                           that will perform the Action
    int indexKernelThread = (blockIdx.x * blockDim.x) + threadIdx.x;

    // To guarantee that not all the Threads will be perform the action, since that for sometimes,
    // it's necessary more Threads than filled positions in the Arrays of Integers
    if(indexKernelThread < size_array) {
        c[indexKernelThread] = a[indexKernelThread] + b[indexKernelThread];
    }

    // Block until all Threads in the block have written their data to shared memory
    // (a Threads' barrier - not necessary, in this case)
    //__syncthreads();
}

/**
 *
 * 2nd step of operations performed by the GPU CUDA's Kernels:
 *
 * The GPU CUDA's Kernel to perform a decrement of an Array of Integers, previously calculated by
 * a sum of all the two Integers contained in two Arrays, performing the computing in parallel.
 *
 * @param c a pointer to the 3rd Array of Integers,
 *          where will be decrement the Integer value contained in each position of the Array
 * @param size_array the total size of the Arrays of Integers, previously allocated
 *
 */
__global__ void decKernel(int* c, const unsigned int size_array) {

    // Index of Kernel/Thread = (Number of Blocks x Size of Block) + Offset associated to the Thread,
    //                           that will perform the Action
    int indexKernelThread = (blockIdx.x * blockDim.x) + threadIdx.x;

    // To guarantee that not all the Threads will be perform the action, since that for sometimes,
    // it's necessary more Threads than filled positions in the Arrays of Integers
    if(indexKernelThread < size_array) {
        c[indexKernelThread]--;
    }

    // Block until all Threads in the block have written their data to shared memory
    // (a Threads' barrier - not necessary, in this case)
    //__syncthreads();
}

/**
 *
 * 3rd step of operations performed by the GPU CUDA's Kernels:
 *
 * The GPU CUDA's Kernel to perform a reversal of an Array of Integers, previously calculated by
 * a decrement of all the Integers contained in an Array, performing the computing in parallel.
 *
 * @param c a pointer to the 3rd Array of Integers
 * @param d a pointer to the 4th Array of Integers,
 *          where will be reversed all the Integer values' positions contained in
 *          each position of the 3rd Array of Integers
 * @param size_array the total size/number of positions of the Arrays of Integers, previously allocated
 *
 */
__global__ void reverseKernel(const int* c, int* d, const unsigned int size_array) {

    // Index of Kernel/Thread = (Number of Blocks x Size of Block) + Offset associated to the Thread,
    //                                  that will perform the Action
    int indexKernelThread = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(indexKernelThread < (size_array / 2)) {

        // Perform the necessary exchanges between the positions of the Array of Integers,
        // in order to reverse all the content of the
        d[size_array - indexKernelThread - 1] = c[indexKernelThread];
        d[indexKernelThread] = c[size_array - indexKernelThread - 1];
    }
    else if (indexKernelThread == (size_array / 2)) {

        // In the case of the size of the Array of Integers is an odd number
        if((size_array % 2) > 0) {

            // It's not necessary to exchange the position of the middle of the Array of Integers,
            // the content of this position will be the same in both Array of Integers
            d[indexKernelThread] = c[indexKernelThread];
        }
    }

    // Block until all Threads in the block have written their data to shared memory
    // (a Threads' barrier - not necessary, in this case)
    //__syncthreads();
}

/**
 * The Main method to perform all the operations by the GPU CUDA's Kernels.
 *
 * @return 0
 */
int main() {

    // Size of the Arrays will be for 4 Integers
    constexpr auto size_array = 4;

    // Size in bytes for the Arrays of Integers
    constexpr auto size_in_bytes = size_array * sizeof(int);

    // Start to calculate all the computing time
    cadlabs::timer<> GPUComputingTimer;
    GPUComputingTimer.start();

    // Initialize random seed to generate random integers
    srand(time(NULL));

    // The CPU/RAM memory buffers in the Host for the 4 Arrays of Integers
    // (2 Arrays for the Inputs and 2 Arrays for the Outputs)
    int* host_a;
    int* host_b;
    int* host_c;
    int* host_d;

    // The GPU memory buffers in the Device for the 4 Arrays of Integers
    // (2 Arrays for the Inputs and 2 Arrays for the Outputs)
    int* dev_a;
    int* dev_b;
    int* dev_c;
    int* dev_d;

    // Allocate CPU/RAM memory buffers in the Host for the 4 Arrays of Integers
    // (2 Arrays for the Inputs and 2 Array for the Output)
    host_a = (int*) malloc(size_in_bytes); // The Array for the Input no. 1 (Array A)
    host_b = (int*) malloc(size_in_bytes); // The Array for the Input no. 2 (Array B)
    host_c = (int*) malloc(size_in_bytes); // The Array for the Output no. 1 (Array C)
    host_d = (int*) malloc(size_in_bytes); // The Array for the Output no. 2 (Array D)

    // Allocate GPU memory buffers in the Device for the 4 Arrays of Integers
    // (2 Arrays for the Inputs and 2 Array for the Output)
    cudaMalloc((void **)&dev_a, size_in_bytes); // The Array for the Input no. 1 (Array A)
    cudaMalloc((void **)&dev_b, size_in_bytes); // The Array for the Input no. 2 (Array B)
    cudaMalloc((void **)&dev_c, size_in_bytes); // The Array for the Output no. 1 (Array C)
    cudaMalloc((void **)&dev_d, size_in_bytes); // The Array for the Output no. 2 (Array D)

    // Check if allocation of GPU memory buffers in the device generated an error
    checkCUDAError("Error in allocation of Devices' memory");

    // Initialising the 1st Array of Integers for the Inputs with random values
    for(int i = 0; i < size_array; i++) {
        host_a[i] = (rand() % 5000);
    }

    // Initialising the 2nd Array of Integers for the Inputs with random values
    for(int i = 0; i < size_array; i++) {
        host_b[i] = (rand() % 5000);
    }

    // Just for debug of the Arrays of Integers for the Inputs
    printf("A = [");
    for(unsigned int i = 0; i < size_array - 1; i++) {
        printf("%d, ", host_a[i]);
    }
    printf("%d]\n", host_a[size_array - 1]);

    printf("B = [");
    for(unsigned int i = 0; i < size_array - 1; i++) {
        printf("%d, ", host_b[i]);
    }
    printf("%d]\n", host_b[size_array - 1]);

    // Copy the Arrays of Integers for the Inputs (Arrays A and B) from the
    // CPU/RAM memory buffers in the Host to the GPU memory buffers of the Device
    cudaMemcpy(dev_a, host_a, size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size_in_bytes, cudaMemcpyHostToDevice);

    // Check if the copy memory of the Host to the memory of the Device generated an error
    checkCUDAError("Error in copy memory of the Host to the memory of the Device");

    // Perform the Sum operation
    addKernel<<< (BLOCK_DIMENSIONS / THREADS_PER_BLOCK),
                  THREADS_PER_BLOCK >>>(dev_a, dev_b, dev_c, size_array);

    // Wait for the kernel to finish,
    // and return any errors encountered during the launch
    cudaDeviceSynchronize();

    // Check if kernel execution generated an error
    checkCUDAError("Error in Kernel's invocation");

    // Copy the Array of Integers for the Output (Array C) from the
    // GPU memory buffers of the Device to the CPU/RAM memory buffers of the Host
    cudaMemcpy(host_c, dev_c, size_in_bytes, cudaMemcpyDeviceToHost);

    // Check if the copy memory of the Device to the memory of the Host generated an error
    checkCUDAError("Error in copy memory of the Device to the memory of the Host");

    // Just for debug of the 3rd Array of Integers
    printf("C = A + B = [");
    for(unsigned int i = 0; i < size_array - 1; i++) {
        printf("%d, ", host_c[i]);
    }
    printf("%d]\n", host_c[size_array - 1]);

    // Perform the Decrement operation
    decKernel<<< (BLOCK_DIMENSIONS / THREADS_PER_BLOCK),
                  THREADS_PER_BLOCK >>>(dev_c, size_array);

    // Wait for the kernel to finish,
    // and return any errors encountered during the launch
    cudaDeviceSynchronize();

    // Copy the Array of Integers for the Output (Array C) from the
    // GPU memory buffers of the Device to the CPU/RAM memory buffers of the Host
    cudaMemcpy(host_c, dev_c, size_in_bytes, cudaMemcpyDeviceToHost);

    // Check if the copy memory of the Device to the memory of the Host generated an error
    checkCUDAError("Error in copy memory of the Device to the memory of the Host");

    // Just for debug of the 3rd Array of Integers
    printf("C - 1 = [");
    for(unsigned int i = 0; i < size_array - 1; i++) {
        printf("%d, ", host_c[i]);
    }
    printf("%d]\n", host_c[size_array - 1]);

    // Perform the Reverse operation
    reverseKernel<<< ((BLOCK_DIMENSIONS / THREADS_PER_BLOCK) / 2),
                       THREADS_PER_BLOCK >>>(dev_c, dev_d, size_array);

    // Wait for the kernel to finish,
    // and return any errors encountered during the launch
    cudaDeviceSynchronize();

    // Copy the Array of Integers for the Output (Array D) from the
    // GPU memory buffers of the Device to the CPU/RAM memory buffers of the Host
    cudaMemcpy(host_d, dev_d, size_in_bytes, cudaMemcpyDeviceToHost);

    // Check if the copy memory of the Device to the memory of the Host generated an error
    checkCUDAError("Error in copy memory of the Device to the memory of the Host");

    // Just for debug of the 4th Array of Integers
    printf("D (Reversal of C) = [");
    for(unsigned int i = 0; i < size_array - 1; i++) {
        printf("%d, ", host_d[i]);
    }
    printf("%d]\n", host_d[size_array - 1]);

    // Free CPU/RAM memory buffers for the 4 Arrays of Integers,
    // previously allocated in Host memory
    // (2 Arrays for the Inputs and 2 Array for the Output)
    free(host_a);
    free(host_b);
    free(host_c);
    free(host_d);

    // Free GPU memory buffers for the 4 Arrays of Integers,
    // previously allocated in Device memory
    // (2 Arrays for the Inputs and 2 Arrays for the Output)
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);

    // Stop to calculate all the computing time
    GPUComputingTimer.stop();

    // Print the metrics of the all computing time
    GPUComputingTimer.print_stats(cout);
    cout << " milliseconds\n ";

    return 0;
}

/**
 *
 * Simple utility function to check for runtime errors in GPU CUDA's Kernels.
 *
 * @param msg the message related to the runtime errors in GPU CUDA's Kernel
 *
 */
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();

    if(cudaSuccess != err) {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err));

        exit(EXIT_FAILURE);
    }
}