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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <pac_helper.h>
using namespace std;


// CPU kernel function to add the elements of two arrays (called vectors)
void add(int* vectorA, int* vectorB, int* vectorC, int size)
{
    //PS: You know how to do this in AVX2, don't you?
    for (int i = 0; i < size; i++)
        vectorC[i] = vectorA[i] + vectorB[i];
}


// Kernel function to add the elements of two arrays using a Grid-Stride loop
__global__ void cudaAdd(int* vectorA, int* vectorB, int* vectorC, int size)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x)
        {
            vectorC[idx] = vectorA[idx] + vectorB[idx]; 
        }
}


int main(void)
{
    // Define the size of the vector: 1048576 elements
    int const N = 1 << 26;

    // Allocate and prepare input/output arrays on host memory
    int* hostVectorA = new int[N];
    int* hostVectorB = new int[N];
    int* hostVectorCCPU = new int[N];
    int* hostVectorCGPU = new int[N];
    for (int i = 0; i < N; i++) {
        hostVectorA[i] = 1;
        hostVectorB[i] = 2;
    }

    // Run the vector kernel on the CPU
    add(hostVectorA, hostVectorB, hostVectorCCPU, N);

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

    // Create structures to hold streams
    const int numStreams = 8;
    cudaStream_t streams[numStreams];

    #pragma omp parallel for
    for (int i = 0; i < numStreams; i++) {
        gpuErrCheck(cudaStreamCreate(&streams[i]));
        int dataOffset = i * (N / numStreams);

        if (i % 2 == 0){
            // launch kernel in stream. On purpose only 1 block, so the kernel takes a long time and we do see the overlap
            cudaAdd <<<1, 1024, 0, streams[i]>>> (&deviceVectorA[dataOffset], &deviceVectorB[dataOffset], &deviceVectorC[dataOffset], N / numStreams);
            gpuErrCheck(cudaPeekAtLastError());
        }else{
            // Using the default Stream! Ouch! On purpose only 1 block, so the kernel takes a long time and we do see the overlap
            cudaAdd <<<1, 1024>>> (&deviceVectorA[dataOffset], &deviceVectorB[dataOffset], &deviceVectorC[dataOffset], N / numStreams);
            gpuErrCheck(cudaPeekAtLastError());
        }
    }

    // Copy the result stored in deviceVectorC back to host (hostVectorCGPU)
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Compare CPU vs GPU result to see if we get the same result
    auto isValid = compareResultVec(hostVectorCCPU, hostVectorCGPU, N);

    // Free memory on device
    gpuErrCheck(cudaFree(deviceVectorA));
    gpuErrCheck(cudaFree(deviceVectorB));
    gpuErrCheck(cudaFree(deviceVectorC));

    // Delete streams
    for (int i = 0; i < numStreams; i++) {
        gpuErrCheck(cudaStreamDestroy(streams[i]));
    }

    // Free memory on host
    delete[] hostVectorA;
    delete[] hostVectorB;
    delete[] hostVectorCCPU;
    delete[] hostVectorCGPU;
}
