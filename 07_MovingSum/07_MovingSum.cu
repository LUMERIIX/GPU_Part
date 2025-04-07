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
    //block has 512Threads
    __shared__ int shm_vec[BLOCKSIZE + 2*RADIUS]; //add 2x radius for the left and right border

    int index = blockIdx.x * blockDim.x + threadIdx.x; //standard line
    shm_vec[threadIdx.x + RADIUS] = vec[index]; //treat 

    if(blockIdx.x != 0 && blockIdx.x != ( WIDTH / BLOCKSIZE ) -1 ) { // if not a border-block load angrenzende elemente (for block idx 1 to n-1)
        if(threadIdx.x < 10 ) { // (left) only execute on threads which need edge blocks!
            for(int i = 0; i < RADIUS; i++) {
                shm_vec[i] = result_vec[index-RADIUS+i];
            }
        }
        else if(threadIdx.x > BLOCKSIZE-RADIUS) {
            for(int i = 0; i < RADIUS; i++) {
                shm_vec[BLOCKSIZE+i] = result_vec[index+RADIUS+i];
            }
        }
    }
    else if(blockIdx.x == 0) { // if first block load angrenzende elemente (for block idx 1 to n-1)
        for(int i = 0; i < RADIUS; i++) {
            shm_vec[i] = 0;
        }
    }
    else if(blockIdx.x == ( WIDTH / BLOCKSIZE ) -1) { // if last block load angrenzende elemente (for block idx 1 to n-1)
        for(int i = 0; i < RADIUS; i++) {
            shm_vec[BLOCKSIZE+i] = 0;
        }
    }
    shm_vec[threadIdx.x + RADIUS] = vec[index]; //treat the current element
    __syncthreads(); //wait for all threads to load their data

    int tmp = 0;
    for(int i = index-RADIUS; i < index+RADIUS+1; i++) { //sum the elements
        tmp += shm_vec[i];
    }

    result_vec[index] = tmp; //write the result


    //ToDo
    //1) create shared cudaMemory
    //2) load data from vec to shm_vec
    //3)!!!attention broder left and right
    //int tmp = 0;
    //for i in readiu;
    //    tmp += shm_vec[i]
//
    //result_vec[i] = tmp
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
__global__ void movingSumSharedMemDynamic(int* vec, int* result_vec, int size)
{
    //ToDo
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
    //ToDo
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
    //ToDo: movingSumSharedMemDynamic <<<nbr_blocks, BLOCKSIZE, ?????????? >>> (deviceVecInput, deviceVecOutput3, WIDTH);
    gpuErrCheck(cudaPeekAtLastError());
    //ToDo: movingSumAtomics << <nbr_blocks, BLOCKSIZE >> > (deviceVecInput, deviceVecOutput4, WIDTH);
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