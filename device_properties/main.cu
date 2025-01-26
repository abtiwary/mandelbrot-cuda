
/**
 * an application to query the device properties and compute capabilities 
 * https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
 */

#include <cstdio>
#include <stdio.h>

int main() {
    int num_devices = 0;

    cudaGetDeviceCount(&num_devices);

    for (int i=0; i < num_devices; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device number: %d\n", i);
        printf("\tDevice name: %s\n", prop.name);
        printf("\tTotal Global Memory: %ld\n", prop.totalGlobalMem);
        printf("\tMemory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("\tMemory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("\tPeak Memory Bandwidth (GB/s): %f\n", 2.0f * prop.memoryClockRate*(prop.memoryBusWidth/8.0f)/1.0e6);
        printf("\n");
        printf("\tDevice major version: %d; Device minor version: %d\n", prop.major, prop.minor);
        printf("\tWarp size: %d\n", prop.warpSize);
        printf("\tMultiprocessor count: %d\n", prop.multiProcessorCount);
        printf("\tMax blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("\tMax threads per multiprocessor: %d\n", prop.maxThreadsPerBlock);
        printf("\tMax blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("\tMax grid size: %d\n", *prop.maxGridSize);
        printf("\tMax threads dims: %d\n", *prop.maxThreadsDim);
    }

    return 0;
}

/*
on my NUC i9 machine with an NVIDIA GTA 1650:

❯ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0

❯ nvcc -o main main.cu
❯ ./main
Device number: 0
        Device name: NVIDIA GeForce GTX 1650
        Memory Clock Rate (KHz): 6001000
        Memory Bus Width (bits): 128
        Peak Memory Bandwidth (GB/s): 192.032000

Device major version: 7; Device minor version: 5
Max blocks per multiprocessor: 16
Max threads per multiprocessor: 1024
Max blocks per multiprocessor: 16
Max grid size: 2147483647
Max threads dims: 1024

*/

