/**
 * A C application that uses CUDA to generate a Mandelbrot image.
 * Designed to work on my NVIDIA GeForce GTX 1650
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

// a structure to represent a complex number
typedef struct _complex {
    float re;
    float im;
} Complex;

// find the complex magnitude
// note: we're not taking the square-root here
__device__ 
float complex_magnitude(Complex* c) {
    return (c->re * c->re) + (c->im * c->im);
}

// add two complex numbers into a given result complex number
__device__ 
void complex_add(Complex* z, Complex* other, Complex* result) {
    result->re = z->re + other->re;
    result->im = z->im + other->im;
}

// multiply two complex numbers into a given result complex number
__device__ 
void complex_multiply(Complex* x, Complex* y, Complex* result) {
    result->re = (x->re * y->re) - (x->im * y->im);
    result->im = (x->re * y->im) + (x->im * y->re);
}

__device__
void get_color(int t, uint8_t* r, uint8_t* g, uint8_t* b) {
    uint8_t palette[][3] = {
        {66, 30, 15},
        {25, 7, 26},
        {9, 1, 47},
        {4, 4, 73},
        {0, 7, 100},
        {12, 44, 138},
        {24, 82, 177},
        {57, 125, 209},
        {134, 181, 229},
        {211, 236, 248},
        {241, 233, 191},
        {248, 201, 95},
        {255, 170, 0},
        {204, 128, 0},
        {153, 87, 0},
        {106, 52, 3},
    };

    int i = t % 16;
    *r = palette[i][0];
    *g = palette[i][1];
    *b = palette[i][2];
}

__global__
void mandelbrot(uint8_t* device_image, int width, int height, int max_iters) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    int index = (y * width + x) * 3;

    if (x < width && y < height) {
        uint8_t r;
        uint8_t g;
        uint8_t b;

        float u = (float)x / height;
        float v = (float)y / height;

        Complex z = {0.0f, 0.0f};
        Complex c = {2.5f * (u - 0.5f) - 1.4f, 2.5f * (v - 0.5f)};

        int i = 0;
        while (i < max_iters && complex_magnitude(&z) < 32.0f) {
            Complex zsq;
            complex_multiply(&z, &z, &zsq);
            complex_add(&zsq, &c, &z);
            i += 1;
        }
        
        float t = (float)i - logf(logf(complex_magnitude(&z)));

        get_color((int)t, &r, &g, &b);

        device_image[index] = r; 
        device_image[index + 1] = g;
        device_image[index + 2] = b; 
    }
}


int main() {
    const int width = 1920;
    const int height = 1080;
    const int max_iterations = 255;

    uint8_t* host_img = (uint8_t*)malloc(width * height * 3);
    if (!host_img) {
        fprintf(stderr, "could not malloc the host image!\n");
        return -1;
    }

    uint8_t* device_img;
    cudaMalloc(&device_img, width * height * 3);
    
    dim3 blocks(32, 32); // max 1024 threads per block on my setup
    dim3 grid(ceil((float)width / 32), ceil((float)height / 32));

    mandelbrot<<<grid, blocks>>>(device_img, width, height, max_iterations);
    cudaDeviceSynchronize();

    cudaMemcpy(host_img, device_img, width * height * 3, cudaMemcpyDeviceToHost);
    
    // write an output image
    FILE* outfile = fopen("/home/pimeson/temp/mandelbrot_cuda.ppm", "w+");
    fprintf(outfile, "P6\n%d %d\n255\n", width, height);
    fwrite(host_img, width * height * 3, sizeof(uint8_t), outfile); 
    fclose(outfile);
    
    free(host_img);
    cudaFree(device_img);

    printf("wrote an image\n");

    return 0;
}

/*
```
❯ nvcc -o main main.cu

// for debug using cuda-gdb
❯ nvcc -g -G -o main main.cu
```
*/

