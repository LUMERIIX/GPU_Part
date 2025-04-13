/*
* This program will calculate a moving sum over an vector, where each output is
* the accumulation of a sliding window. In other words, the input for each
* output is defined by the same global index but also accumulates the items left
* and right of it of the size RADIUS.
*
*    ┌─────────────────────────────┐
*    │1 1 1 1 1 1 1 1 1 1 1 1 1 1 1│
*    └▲─▲─────▲─▲─▲───▲─▲─▲────────┘
*     │ │     │ │ │   │ │ │
*     ├─┘     └─┼─┘   └─┼─┘
*     │         │       │-->RADIUS=1
*     │         │       │
*     │         │       │
*    ┌┴─────────┴───────┴───────────┐
*    │2 3 3 3 3 3 3 3 3 3 3 3 3 3 2 │
*    └──────────────────────────────┘
*    * Only three sliding window are drawn here, not all of them
*
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>

using namespace std;

#define RADIUS      10
#define BLOCKSIZE   512
#define WIDTH       65536

/* This is our CUDA call wrapper, we will use in PAC.
*
*  Almost all CUDA calls should be wrapped with this makro.
*  Errors from these calls will be catched and printed on the console.
*  If an error appears, the program will terminate.
*
* Example: gpuErrCheck(cudaMalloc(&deviceA, N * sizeof(int)));
*          gpuErrCheck(cudaMemcpy(deviceA, hostA, N * sizeof(int), cudaMemcpyHostToDevice));
*/
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
        {
            exit(code);
        }
    }
}


/*
*
* Rewrite the kernel movingSumGlobal using a static allocation of shared
* memory, as shown in the slides of the lecture.
*
*/
__global__ void movingSumSharedMemStatic(int* vec, int* result_vec, int size) //size = 10.. so sum includes i-10.....i....i+10
{
    // Assuming BLOCKSIZE and RADIUS are defined appropriately.
    __shared__ int shm_vec[BLOCKSIZE + 2 * RADIUS];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.x + RADIUS; // Position in shared memory

    // Load the main data element for the current thread
    if(index < size) {
        shm_vec[local_index] = vec[index];

        if(threadIdx.x < RADIUS) { //use the n-threads (RADIUIS ones) to handle left and right borders
            // left halo handling
            if( index < RADIUS ) { // thread all to the left (there are no left neighbours)
                shm_vec[threadIdx.x] = 0;
            }
            else {
                shm_vec[threadIdx.x] = vec[index - RADIUS];
            }

            //right halo handling
            if (index + BLOCKSIZE >= size) { //if last block there are no neighbours
                shm_vec[BLOCKSIZE + local_index] = 0;
            }
            else {
                shm_vec[BLOCKSIZE + local_index] = vec[index + BLOCKSIZE];
            }
        }
    }
    
    __syncthreads(); // Ensure all threads have loaded data into shared memory

    // Compute the moving sum using the shared memory indices
    int tmp = 0;
    for (int i = local_index - RADIUS; i <= local_index + RADIUS; i++) {
        tmp += shm_vec[i];
    }
    
    // Store the result, ensuring we don't write out-of-bounds
    if(index < size)
        result_vec[index] = tmp;
}


/*
*
* Rewrite the kernel movingSumGlobal using a dynamic allocation of shared
* memory, more information can be found here:
* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
*
* You should use something like this:
*    extern __shared__ int shmVec[];
* and extend the execution configuration with the size of the shmVec.
*
<<< n_blocks, n_threads, size shm >>>
annahme:
- threads 1024
- shm muss links und rechts +10 elemente haben 
*/
/**
 * Threads that belong to a thread block cluster, can read, write or perform atomics in the distributed address space, regardless whether the address belongs to the local thread block or a remote thread block. 
 * Whether a kernel uses distributed shared memory or not, the shared memory size specifications, static or dynamic is still per thread block. The size of distributed shared memory is just the number of thread blocks per cluster multiplied by the size of shared memory per thread block. 
 **/
__global__ void movingSumSharedMemDynamic(int* vec, int* result_vec, int size)
{
    extern __shared__ int shm_vec[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.x + RADIUS; // Position in shared memory

    // Load the main data element for the current thread
    if(index < size) {
        shm_vec[local_index] = vec[index];

        if(threadIdx.x < RADIUS) { //use the n-threads (RADIUIS ones) to handle left and right borders
            // left halo handling
            if( index < RADIUS ) { // thread all to the left (there are no left neighbours)
                shm_vec[threadIdx.x] = 0;
            }
            else {
                shm_vec[threadIdx.x] = vec[index - RADIUS];
            }

            //right halo handling
            if (index + BLOCKSIZE >= size) { //if last block there are no neighbours
                shm_vec[BLOCKSIZE + local_index] = 0;
            }
            else {
                shm_vec[BLOCKSIZE + local_index] = vec[index + BLOCKSIZE];
            }
        }
    }
    
    __syncthreads(); // Ensure all threads have loaded data into shared memory

    // Compute the moving sum using the shared memory indices
    int tmp = 0;
    for (int i = local_index - RADIUS; i <= local_index + RADIUS; i++) {
        tmp += shm_vec[i];
    }
    
    // Store the result, ensuring we don't write out-of-bounds
    if(index < size)
        result_vec[index] = tmp;
}


/*
*
* Rewrite the kernel movingSumGlobal using only global memory and no
* shared mem. Use atomic add operations.
* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
*
* You must reverse the access pattern, such that without atomics, conflicts can occure.
* So 1 Thread writes "its" value into multiple outputs.
*/
__global__ void movingSumAtomics(int* vec, int* result_vec, int size)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int offset = -RADIUS; offset <= RADIUS; offset++) { //a thread writes to all its neighbours his additional value
        int internalIdx = globalIdx + offset;
        if (internalIdx >= 0 && internalIdx < size) {
            atomicAdd(result_vec + internalIdx, vec[globalIdx]);
        }
    }
}


// This is the GPU refernece implementation
__global__ void movingSumGlobal(int* vec, int* result_vec, int size)
{

    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int result = 0;
    if (globalIdx >= RADIUS && globalIdx < size - RADIUS) {
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += vec[globalIdx + offset];
        }
    }

    if (globalIdx < RADIUS) {
        for (int offset = 0 - globalIdx; offset <= RADIUS; offset++) {
            result += vec[globalIdx + offset];
        }
    }

    if (globalIdx < size && globalIdx >= size - RADIUS) {
        for (int offset = -RADIUS; offset < size - globalIdx; offset++) {
            result += vec[globalIdx + offset];
        }
    }

    result_vec[globalIdx] = result;
}


// CPU reference implementation
void movingSumCPU(int* vec, int* result_vec, int size)
{
    int result;

    for (int i = 0; i < size; ++i) {
        result = 0;

        if (i >= RADIUS && i < size - RADIUS) {
            for (int offset = -RADIUS; offset <= RADIUS; offset++) {
                result += vec[i + offset];
            }
        }

        if (i < RADIUS) {
            for (int offset = 0 - i; offset <= RADIUS; offset++) {
                result += vec[i + offset];
            }
        }

        if (i < size && i >= size - RADIUS) {
            for (int offset = -RADIUS; offset < size - i; offset++) {
                result += vec[i + offset];
            }
        }

        result_vec[i] = result;
    }
}


// Compare result vectors
int compareResultVec(int* vectorCPU, int* vectorGPU, int size)
{
    int error = 0;
    for (int i = 0; i < size; i++)
    {
        error += abs(vectorCPU[i] - vectorGPU[i]);
    }
    if (error == 0)
    {
        cout << "No errors. All good!" << endl;
        return 0;
    }
    else
    {
        cout << "Accumulated error: " << error << endl;
        return -1;
    }
}


int main(void)
{
    // Allocate and prepare input vector on host memory
    int* hostVecInput = new int[WIDTH];
    int* hostVecOutputCPU = new int[WIDTH];
    int* hostVecOutputGPU1 = new int[WIDTH];
    int* hostVecOutputGPU2 = new int[WIDTH];
    int* hostVecOutputGPU3 = new int[WIDTH];
    int* hostVecOutputGPU4 = new int[WIDTH];

    for (int i = 0; i < WIDTH; i++) {
        hostVecInput[i] = 1;
    }

    // Get the CPU result
    movingSumCPU(hostVecInput, hostVecOutputCPU, WIDTH);

    // Allocate device memory
    int* deviceVecInput;
    int* deviceVecOutput1;
    int* deviceVecOutput2;
    int* deviceVecOutput3;
    int* deviceVecOutput4;
    gpuErrCheck(cudaMalloc(&deviceVecInput, WIDTH * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVecOutput1, WIDTH * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVecOutput2, WIDTH * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVecOutput3, WIDTH * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVecOutput4, WIDTH * sizeof(int)));

    // Copy data from host to device
    gpuErrCheck(cudaMemcpy(deviceVecInput, hostVecInput, WIDTH * sizeof(int), cudaMemcpyHostToDevice));

    // Run kernel on all elements on the GPU
    int nbr_blocks = ((WIDTH % BLOCKSIZE) != 0) ? (WIDTH / BLOCKSIZE + 1) : (WIDTH / BLOCKSIZE);
    //WIDTH = array-elems
    movingSumGlobal << <nbr_blocks, BLOCKSIZE >> > (deviceVecInput, deviceVecOutput1, WIDTH);
    gpuErrCheck(cudaPeekAtLastError());
    movingSumSharedMemStatic << <nbr_blocks, BLOCKSIZE >> > (deviceVecInput, deviceVecOutput2, WIDTH);
    gpuErrCheck(cudaPeekAtLastError());
    //configuration how much shared memory is needed is passed in the kernel call << nbr_blocks, BLOCKSIZE, size shmVec >
    size_t shmSize = (BLOCKSIZE + 2 * RADIUS) * sizeof(int);
    movingSumSharedMemDynamic <<<nbr_blocks, BLOCKSIZE, shmSize >>> (deviceVecInput, deviceVecOutput3, WIDTH);
    gpuErrCheck(cudaPeekAtLastError());
    movingSumAtomics << <nbr_blocks, BLOCKSIZE >> > (deviceVecInput, deviceVecOutput4, WIDTH);
    gpuErrCheck(cudaPeekAtLastError());

    // Copy the result stored in device_y back to host_y
    gpuErrCheck(cudaMemcpy(hostVecOutputGPU1, deviceVecOutput1, WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrCheck(cudaMemcpy(hostVecOutputGPU2, deviceVecOutput2, WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrCheck(cudaMemcpy(hostVecOutputGPU3, deviceVecOutput3, WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrCheck(cudaMemcpy(hostVecOutputGPU4, deviceVecOutput4, WIDTH * sizeof(int), cudaMemcpyDeviceToHost));

    // Check for errors in result
    auto ret = compareResultVec(hostVecOutputCPU, hostVecOutputGPU1, WIDTH);
    ret = compareResultVec(hostVecOutputCPU, hostVecOutputGPU2, WIDTH);
    ret = compareResultVec(hostVecOutputCPU, hostVecOutputGPU3, WIDTH);
    ret = compareResultVec(hostVecOutputCPU, hostVecOutputGPU4, WIDTH);

    // Free memory on device & host
    cudaFree(deviceVecInput);
    cudaFree(deviceVecOutput1);
    cudaFree(deviceVecOutput2);
    cudaFree(deviceVecOutput3);
    cudaFree(deviceVecOutput4);
    delete[] hostVecInput;
    delete[] hostVecOutputCPU;
    delete[] hostVecOutputGPU1;
    delete[] hostVecOutputGPU2;
    delete[] hostVecOutputGPU3;
    delete[] hostVecOutputGPU4;

    return 0;
}