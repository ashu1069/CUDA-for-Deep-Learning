#include <cstdio>
#include <cuda_runtime.h>

// THE vectorAdd KERNEL
// global marks this as a GPU kernel callable from the CPU. 
// The pointers a, b, and c will point to GPU memory.
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Each thread gets its unique ID from threadIdx.x to determine which element to process.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Only process elements up to n (boundary check prevents out-of-bounds memory access when thread count exceeds data size)
    if (i < n) {        // Check if thread index is within valid range for the array
        // Perform the work only if this thread has valid data to process.
        c[i] = a[i] + b[i];
    }
}

// HOST MEMORY LOCATION
int main() {
    int n = 8;
    // Calculate total bytes needed: 8 floats * 4 bytes per float = 32 bytes
    size_t bytes = n * sizeof(float);

    // malloc (bytes) allocates memory and returns a pointer. The h_ prefix is a convention meaning "host" memory.
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // INITIALIZE HOST DATA
    for (int i =0; i < n; i++) {
        h_a[i] = (float)i;  // h_a = [0, 1, 2, 3, 4, 5, 6, 7]
        h_b[i] = (float)(i * 2); // h_b = [0, 2, 4, 6, 8, 10, 12, 14]
    }

    // DEVICE MEMORY ALLOCATION
    // The d_ prefix is a convention meaning "device" memory. These pointers will hold GPU memory addresses
    float *d_a, *d_b, *d_c;
    // cudaMalloc allocates GPU memory. The (void**)&d_a syntax passes the address of our pointer so cudaMalloc can fill it in
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // COPY DATA TO DEVICE
    // cudaMemcpy(destination, source, size, direction) copies data. 
    // The direction flag cudaMemcpyHostToDevice specifies data is coming from host(CPU) to device(GPU).
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // LAUNCH THE KERNEL
    // <<<grid_size, block_size>>> specifies how many blocks and threads to launch.
    // Here, we launch 1 block with 8 threads.
    vectorAdd<<<1, 8>> >(d_a, d_b, d_c, n);

    // COPY DATA BACK TO HOST
    // cudaMemcpy(destination, source, size, direction) copies data. 
    // The direction flag cudaMemcpyDeviceToHost specifies data is coming from device(GPU) to host(CPU).
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // VERIFICATION AND CLEANUP
    int success = 1;
    for (int i =0; i < n; ++i) {
        if (h_c[i] != (h_a[i] + h_b[i])) {
            printf("Error at index %d: Got %f ,expected %f\n", i, h_c[i], h_a[i] + h_b[i]);
            success = 0;
            break;
        }
    }

    if (success) {
        printf("All elements are correct\n");
    }
    
    // free() releases host memory allocated with malloc()
    free(h_a);
    free(h_b);
    free(h_c);
    // cudaFree() releases device memory allocated with cudaMalloc()
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}