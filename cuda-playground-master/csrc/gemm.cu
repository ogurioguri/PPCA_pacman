#include <cassert>
#include <chrono>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <random>

// You may increase this value to test larger matrices
// But it will be slow on CPU
constexpr int MAXN = 2048;

/**
 * @brief A naive implementation of matrix multiplication on CPU.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */
void naiveSgemm(float *a, float *b, float *c, const int M, const int N,
                const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += a[m * K + k] * b[k * N + n];
      }
      c[m * N + n] = sum;
    }
  }
}

/**
 * @brief A naive implementation of matrix multiplication on GPU.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */

__global__ void naiveSgemm2D(float *a, float *b, float *c, const int M,
                             const int N, const int K) {
  int block_size = 32 *32;
  int row = blockIdx.y * blockDim.y * 2 + threadIdx.y;
  int col = blockIdx.x * blockDim.x  + threadIdx.x;
  float val[2] = {0.0f};
  __shared__ float s_a[block_size][block_size];
  __shared__ float s_b[block_size][block_size];


  int iter = (K + block_size - 1) / block_size;
  //a fetch two members , b fetch one member
  for (int i = 0; i < iter; i++) {
    s_a[threadIdx.y][threadIdx.x] = a[row * K + i * block_size + threadIdx.x];
    s_a[threadIdx.y + 16][threadIdx.x] = a[(row + 16) * K + i * block_size + threadIdx.x];
    s_b[threadIdx.y][threadIdx.x] = b[(i * block_size + threadIdx.y) * N + col];
    s_b[threadIdx.y + 16][threadIdx.x] = b[(i * block_size + threadIdx.y + 16) * N + col];
    __syncthreads();

    for (int j = 0; j < block_size; j++) {
      val[0] += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
      val[1] += s_a[threadIdx.y +16][j] * s_b[j][threadIdx.x];
    }
    __syncthreads();
  }
  c[row * N + col] = val[0];
  c[(row + 16) * N + col] = val[1];

  // transfer first tile from global mem to shared mem
  


}

/**
 * @brief Launch naiveSgemm2D kernel.
 */
void launchSgemm2D(float *a, float *b, float *c, const int M, const int N,
                   const int K) {
  dim3 block(16, 16); // 256 threads per block (16 * 16 = 256)
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  naiveSgemm2D<<<grid, block>>>(a, b, c, M, N, K);
}

void initialize(float *a, float *b, float *c, const int M, const int N,
                const int K) {
  auto gen = std::mt19937(2024);
  auto dis = std::uniform_real_distribution<float>(-1.0, 1.0);
  for (int i = 0; i < M * K; ++i) {
    a[i] = dis(gen);
  }
  for (int i = 0; i < K * N; ++i) {
    b[i] = dis(gen);
  }
  for (int i = 0; i < M * N; ++i) {
    c[i] = 0.0;
  }
}

/** 
 * @brief Launch sgemm using cuBLAS
 */
void launchCublasSgemm(float *a, float *b, float *c, const int M, const int N,
                       const int K) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K,
              &beta, c, N);
}


int main() {
  float *a, *b, *c;
  a = new float[MAXN * MAXN];
  b = new float[MAXN * MAXN];
  c = new float[MAXN * MAXN];
  initialize(a, b, c, MAXN, MAXN, MAXN);

  // ********** CPU **********
  auto start = std::chrono::high_resolution_clock::now();
  naiveSgemm(a, b, c, MAXN, MAXN, MAXN);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  printf("CPU time: %.3fs\n", elapsed.count());

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, MAXN * MAXN * sizeof(float));
  cudaMalloc(&d_b, MAXN * MAXN * sizeof(float));
  cudaMalloc(&d_c, MAXN * MAXN * sizeof(float));
  cudaMemcpy(d_a, a, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);

  // ********** GPU **********
  start = std::chrono::high_resolution_clock::now();
  launchSgemm2D(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  printf("GPU time: %.3fs\n", elapsed.count());

  // ********** cuBLAS **********
  start = std::chrono::high_resolution_clock::now();
  launchCublasSgemm(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  printf("cuBLAS time: %.3fs\n", elapsed.count());
}
