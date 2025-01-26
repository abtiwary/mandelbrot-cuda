
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
        
        printf("\n\t-----[General Information]-----\n");
        printf("\tDevice name: %s\n", prop.name);
        printf("\t(Compute capability) Device major version: %d; Device minor version: %d\n", prop.major, prop.minor);
        printf("\tClock rate: %d\n", prop.clockRate);

        printf("\n\t-----[Memory Information for `device`]-----\n");
        printf("\tTotal Global Memory: %ld\n", prop.totalGlobalMem);
        printf("\tTotal Constant Memory: %ld\n", prop.totalConstMem);
        printf("\tMemory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("\tMemory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("\tPeak Memory Bandwidth (GB/s): %f\n", 2.0f * prop.memoryClockRate*(prop.memoryBusWidth/8.0f)/1.0e6);

        printf("\n\t-----[MP Information for `device`]-----\n");
        printf("\tMultiprocessor count: %d\n", prop.multiProcessorCount);
        printf("\tShared memory per MP: %ld\n", prop.sharedMemPerBlock);
        printf("\tRegisters per MP: %d\n", prop.regsPerMultiprocessor);
        printf("\tRegisters per block: %d\n", prop.regsPerBlock);
        printf("\tThreads in warp (warp size): %d\n", prop.warpSize);
        printf("\tMax threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("\tMax blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("\tMax grid dims: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\tMax threads dims: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

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

        -----[General Information]-----
        Device name: NVIDIA GeForce GTX 1650
        (Compute capability) Device major version: 7; Device minor version: 5
        Clock rate: 1590000

        -----[Memory Information for `device`]-----
        Total Global Memory: 4086169600
        Total Constant Memory: 65536
        Memory Clock Rate (KHz): 6001000
        Memory Bus Width (bits): 128
        Peak Memory Bandwidth (GB/s): 192.032000

        -----[MP Information for `device`]-----
        Multiprocessor count: 14
        Shared memory per MP: 49152
        Registers per MP: 65536
        Registers per block: 65536
        Threads in warp (warp size): 32
        Max threads per block: 1024
        Max blocks per multiprocessor: 16
        Max grid dims: (2147483647, 65535, 65535)
        Max threads dims: (1024, 1024, 64)

*/

