#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>

using namespace std;
namespace fs = std::filesystem;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while (0)

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Read square matrix into a flat 1D vector (row-major) for CUDA compatibility
bool readMatrixFlat(const string& filepath, vector<double>& matrix, int& n) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filepath << endl;
        return false;
    }

    if (!(file >> n) || n <= 0) {
        cerr << "Invalid matrix size in file: " << filepath << endl;
        return false;
    }

    matrix.resize(static_cast<size_t>(n) * n);
    for (int i = 0; i < n * n; ++i) {
        if (!(file >> matrix[i])) {
            cerr << "Error reading matrix elements from " << filepath << endl;
            return false;
        }
    }
    file.close();
    return true;
}

// Write flat 1D vector as a 2D square matrix to a file
bool writeMatrixFlat(const string& filepath, const vector<double>& matrix, int n) {
    fs::path p(filepath);
    if (p.has_parent_path() && !fs::exists(p.parent_path())) {
        fs::create_directories(p.parent_path());
    }

    ofstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening file for writing: " << filepath << endl;
        return false;
    }

    file << n << "\n";
    file << fixed << setprecision(6);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[static_cast<size_t>(i) * n + j] << (j == n - 1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
    return true;
}

int main(int argc, char* argv[]) {
    // Arguments: A.txt B.txt C.txt [block_size]
    string fileA = (argc > 1) ? argv[1] : "A.txt";
    string fileB = (argc > 2) ? argv[2] : "B.txt";
    string fileC = (argc > 3) ? argv[3] : "C.txt";
    int blockSize = (argc > 4) ? stoi(argv[4]) : 16;

    if (blockSize <= 0 || blockSize > 32) {
        cerr << "Error: block_size must be between 1 and 32 (max 1024 threads per block)." << endl;
        return 1;
    }

    vector<double> h_A, h_B;
    int nA = 0, nB = 0;

    if (!readMatrixFlat(fileA, h_A, nA) || !readMatrixFlat(fileB, h_B, nB)) {
        return 1;
    }

    if (nA != nB) {
        cerr << "Error: Matrix dimensions do not match (" << nA << " != " << nB << ")" << endl;
        return 1;
    }

    int N = nA;
    size_t bytes = static_cast<size_t>(N) * N * sizeof(double);
    vector<double> h_C(static_cast<size_t>(N) * N, 0.0);

    // Allocate device memory
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // Configure 2D grid and block dimensions
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    // Measure pure kernel execution time using CUDA Events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Cleanup GPU resources
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    double total_flops = 2.0 * N * N * N - N * N;

    if (!writeMatrixFlat(fileC, h_C, N)) {
        return 1;
    }

    cout << "\n--- CUDA EXECUTION RESULTS ---" << endl;
    cout << "Matrix Dimension (N): " << N << "x" << N << endl;
    cout << "CUDA Block Size: " << blockSize << "x" << blockSize
         << " (" << blockSize * blockSize << " threads/block)" << endl;
    cout << "CUDA Grid Size: " << numBlocks.x << "x" << numBlocks.y << " blocks" << endl;
    cout << "Total Elements: " << N * N << endl;
    cout << "Workload (FLOP): " << fixed << setprecision(0) << total_flops << endl;
    cout << "Execution Time: " << fixed << setprecision(3) << kernel_ms << " ms" << endl;
    cout << "Output File Saved: " << fileC << endl;

    return 0;
}
