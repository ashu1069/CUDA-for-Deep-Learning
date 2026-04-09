#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

// CPU TRANSPOSE (reference on host)
// Row-major input: element (row, col) lives at in[row * num_cols + col].
// Output is also stored row-major but with dimensions swapped, so (row,col) maps to out[col * num_rows + row].
void transpose_cpu(const float* in, float* out, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
            out[col * num_rows + row] = in[row * num_cols + col];
        }
    }
}

// THE transpose_kernel
// __global__ marks this as a GPU kernel callable from the CPU.
// The pointers in and out will point to GPU memory.
__global__ void transpose_kernel(const float* in, float* out, int num_rows, int num_cols) {
    // 2D grid: each thread is identified by (row, col) in the matrix.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check prevents out-of-bounds access when the thread grid covers a padded rectangle.
    if (row < num_rows && col < num_cols) {
        int in_index = row * num_cols + col;
        int out_index = col * num_rows + row;
        out[out_index] = in[in_index];
    }
}

// Compare two host buffers elementwise (used to check GPU output against CPU).
static bool verify_equal(const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            printf("Mismatch at %zu: got %f expected %f\n", i, b[i], a[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int num_rows = 4096;
    const int num_cols = 4096;
    const size_t n_in = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols);
    // Total bytes: rows * cols floats, 4 bytes per float.
    const size_t bytes_in = n_in * sizeof(float);

    // HOST MEMORY
    // malloc(bytes) allocates CPU memory. The h_ prefix means "host".
    float* h_in = (float*)malloc(bytes_in);
    float* h_out_cpu = (float*)malloc(bytes_in);
    float* h_out_gpu = (float*)malloc(bytes_in);
    if (!h_in || !h_out_cpu || !h_out_gpu) {
        printf("Host malloc failed\n");
        return 1;
    }

    // INITIALIZE HOST DATA
    for (size_t i = 0; i < n_in; ++i) {
        h_in[i] = static_cast<float>(i % 997);
    }

    transpose_cpu(h_in, h_out_cpu, num_rows, num_cols);

    // DEVICE MEMORY ALLOCATION
    // The d_ prefix means "device". cudaMalloc fills in GPU pointers; (void**)&d_in passes the address of the pointer.
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_in, bytes_in);
    cudaMalloc((void**)&d_out, bytes_in);

    // COPY DATA TO DEVICE
    // cudaMemcpy(destination, source, size, direction). cudaMemcpyHostToDevice = CPU -> GPU.
    cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice);

    // LAUNCH CONFIGURATION
    // 2D blocks of threads; grid size is chosen so every matrix element gets a thread (with boundary checks).
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // LAUNCH THE KERNEL (correctness run)
    // <<<blocks, threads>>> is the CUDA launch configuration for a 2D grid.
    transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
    cudaDeviceSynchronize();

    // COPY DATA BACK TO HOST
    // cudaMemcpyDeviceToHost = GPU -> CPU.
    cudaMemcpy(h_out_gpu, d_out, bytes_in, cudaMemcpyDeviceToHost);

    if (!verify_equal(h_out_cpu, h_out_gpu, n_in)) {
        printf("CPU and GPU results differ.\n");
        free(h_in);
        free(h_out_cpu);
        free(h_out_gpu);
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }
    printf("Correctness: CPU and GPU match.\n");

    // BENCHMARK PARAMETERS
    const int warmup = 3;
    const int iters = 20;

    // CPU TIMING (host only: data already in h_in / h_out_cpu)
    for (int w = 0; w < warmup; ++w) {
        transpose_cpu(h_in, h_out_cpu, num_rows, num_cols);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        transpose_cpu(h_in, h_out_cpu, num_rows, num_cols);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(iters);

    // GPU TIMING — KERNEL ONLY
    // cudaEventRecord pairs measure elapsed GPU time between markers (milliseconds).
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int w = 0; w < warmup; ++w) {
        transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_kernel_ms = 0.f;
    cudaEventElapsedTime(&gpu_kernel_ms, start, stop);
    gpu_kernel_ms /= static_cast<float>(iters);

    // GPU TIMING — END TO END (H2D + kernel + D2H each iteration, like a fresh transfer every time)
    for (int w = 0; w < warmup; ++w) {
        cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice);
        transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
        cudaMemcpy(h_out_gpu, d_out, bytes_in, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice);
        transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
        cudaMemcpy(h_out_gpu, d_out, bytes_in, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_e2e_ms = 0.f;
    cudaEventElapsedTime(&gpu_e2e_ms, start, stop);
    gpu_e2e_ms /= static_cast<float>(iters);

    printf("Matrix %d x %d, %d iters (warmup %d)\n", num_rows, num_cols, iters, warmup);
    printf("CPU:              %.3f ms/iter\n", cpu_ms);
    printf("GPU kernel only:  %.3f ms/iter\n", gpu_kernel_ms);
    printf("GPU H2D+kernel+D2H: %.3f ms/iter\n", gpu_e2e_ms);
    if (cpu_ms > 0.0) {
        printf("Speedup vs CPU — kernel: %.2fx, e2e: %.2fx\n", cpu_ms / gpu_kernel_ms,
               cpu_ms / static_cast<double>(gpu_e2e_ms));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // free() releases host memory; cudaFree() releases device memory.
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
