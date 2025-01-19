# Mandelbrot set generator using CUDA

This is a CUDA kernel that generates and writes out an image containing a Mandelbrot set.

# Build 

### My setup 

My dev machine is an i9 NUC with a discrete `NVIDIA GeForce GTX 1650` card.

Some device properties:
```
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
```

## Build instructions

The build process is relatively simple, once the CUDA toolchain is installed. 

On my Kubuntu machine, i had to sudo apt install `nvidia-cuda-toolkit` (and of course the appropriate
driver).

`nvcc` can then be used to build the kernel:

```
nvcc -o main main.cu
```


