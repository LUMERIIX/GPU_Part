/* This program will do a vector addition on two vecotrs.
*  They have the same size N (defined in main).
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
*/
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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


// CPU kernel function to add the elements of two arrays (called vectors)
void add(int* vectorA, int* vectorB, int* vectorC, int size)
{
    //PS: You know how to do this in AVX2, don't you?
    for (int i = 0; i < size; i++)
        vectorC[i] = vectorA[i] + vectorB[i];
}


// Kernel function to add the elements of two arrays
__global__ void cudaAdd(int* vectorA, int* vectorB, int* vectorC, int size)
{
    int globThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globThreadId < size) {
        vectorC[globThreadId] = vectorA[globThreadId] + vectorB[globThreadId];
    }
    else{
        printf("Overshooted");
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
    gpuErrCheck(cudaMalloc(&deviceVectorA, N * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVectorB, N * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVectorC, N * sizeof(int)));

    // Copy data from host to device
    gpuErrCheck(cudaMemcpy(deviceVectorA, hostVectorA, N * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpy(deviceVectorB, hostVectorB, N * sizeof(int), cudaMemcpyHostToDevice));

    // Run the vector kernel on the CPU
    add(hostVectorA, hostVectorB, hostVectorCCPU, N);

    int configs[5][2] = {{1024,1024},{1024,1024},{2048,512},{4096,256},{8192,128}}; //first is warm-up

    // Run kernel on the GPU
    // ToDo: Play with different block/thread sizes - do you see significant differences?
    //       1048576 Threads are needed to have 1 Thread per addition
    for(int i = 0; i < 5;i++) {
        cudaAdd << <configs[i][0], configs[i][1] >> > (deviceVectorA, deviceVectorB, deviceVectorC, N);// Kernel execution is async and will not return an error:
        gpuErrCheck(cudaPeekAtLastError());
    }

    // Copy the result stored in deviceVectorC back to host (hostVectorCGPU)
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Compare CPU vs GPU result to see if we get the same result
    auto isValid = compareResultVec(hostVectorCCPU, hostVectorCGPU, N);

    // Free memory on device
    gpuErrCheck(cudaFree(deviceVectorA));
    gpuErrCheck(cudaFree(deviceVectorB));
    gpuErrCheck(cudaFree(deviceVectorC));

    // Free memory on host
    delete[] hostVectorA;
    delete[] hostVectorB;
    delete[] hostVectorCCPU;
    delete[] hostVectorCGPU;
}