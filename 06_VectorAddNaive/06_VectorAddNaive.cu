/* This program will transfer data to and from the GPU.
*  It will do a vector addition on two vecotrs.
*  They have the same size N (defined in main).
*
*
*  +---------+   +---------+   +---------+
*  |111111111| + |222222222| = |333333333|
*  +---------+   +---------+   +---------+
*
*  vectorA   = all Ones
*  vectorB   = all Twos
*  vectorC   = all Three
*
*  NOTE: vectorX is an array of int and not std::vector
*
*  It should help to understand how a processing flow with a GPU
*  works. When we have to transfer data and which resources are
*  used therefore.
*
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;


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


// Compare result arrays CPU vs GPU result. If no diff, the result pass.
int compareResultVec(int* vectorCPU, int* vectorGPU, int size)
{
    int error = 0;
    for (int i = 0; i < size; i++)
    {
        error += abs(vectorCPU[i] - vectorGPU[i]);
    }
    if (error == 0)
    {
        cout << "Test passed." << endl;
        return 0;
    }
    else
    {
        cout << "Accumulated error: " << error << endl;
        return -1;
    }
}


// CPU kernel function to add the elements of two arrays (called vectors)
void add(int* vectorA, int* vectorB, int* vectorC, int size)
{
    //PS: You know how to do this in AVX2, don't you?
    for (int i = 0; i < size; i++)
        vectorC[i] = vectorA[i] + vectorB[i];
}


// GPU kernel function to add the elements of two arrays (called vectors)
__global__ void cudaAdd(int* vectorA, int* vectorB, int* vectorC, int size)
{
    for (int i = 0; i < size; i++)
        vectorC[i] = vectorA[i] + vectorB[i];
}


int main(void)
{
    // Define the size of the vector: 1048576 elements
    int N = 1 << 20;

    // Allocate and prepare input/output arrays on host memory
    int* hostVectorA = new int[N];
    int* hostVectorB = new int[N];
    int* hostVectorCCPU = new int[N];
    int* hostVectorCGPU = new int[N];
    for (int i = 0; i < N; i++) {
        hostVectorA[i] = 1;
        hostVectorB[i] = 2;
    }

    // Alloc N times size of int at device memory for deviceVector[A-C]
    int* deviceVectorA;
    int* deviceVectorB;
    int* deviceVectorC;
    cudaMalloc(&deviceVectorA, N * sizeof(int));
    cudaMalloc(&deviceVectorB, N * sizeof(int));
    cudaMalloc(&deviceVectorC, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(deviceVectorA, hostVectorA,N * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVectorB, hostVectorB,N * sizeof(int),cudaMemcpyHostToDevice);

    // Run the vector kernel on the CPU
    add(hostVectorA, hostVectorB, hostVectorCCPU, N);

    // Run kernel on the GPU
    // INFO: We talk about this in the next lecture
    // deviceVector[A-C] are int* on device memory, which you created above
    cudaAdd <<<1, 1 >>> (deviceVectorA, deviceVectorB, deviceVectorC, N);
    // Kernel execution is async and will not return an error:
    gpuErrCheck(cudaPeekAtLastError());

    // Copy the result stored in deviceVectorC back to host (hostVectorCGPU)
    cudaMemcpy(hostVectorCGPU, deviceVectorC,N * sizeof(int),cudaMemcpyDeviceToHost);

    // Compare CPU vs GPU result to see if we get the same result
    auto isValid = compareResultVec(hostVectorCCPU, hostVectorCGPU, N);

    // Free memory on device
    cudaFree(deviceVectorA);
    cudaFree(deviceVectorB);
    cudaFree(deviceVectorC);

    // Free memory on host
    delete[] hostVectorA;
    delete[] hostVectorB;
    delete[] hostVectorCCPU;
    delete[] hostVectorCGPU;

    return isValid;
}