#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <string>
#include <filesystem>
#include <random>
#include <omp.h>

using namespace std;
namespace fs = std::filesystem;

bool readMatrix(const string& filepath, vector<vector<double>>& matrix, int& n) {
    ifstream file(filepath);
    if (!file.is_open()) return false;
    if (!(file >> n) || n <= 0) return false;

    matrix.assign(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(file >> matrix[i][j])) return false;
        }
    }
    file.close();
    return true;
}

bool writeMatrix(const string& filepath, const vector<vector<double>>& matrix, int n) {
    fs::path p(filepath);
    if (p.has_parent_path() && !fs::exists(p.parent_path())) {
        fs::create_directories(p.parent_path());
    }

    ofstream file(filepath);
    if (!file.is_open()) return false;

    file << n << "\n" << fixed << setprecision(6);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i][j] << (j == n - 1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
    return true;
}

// Random matrix generator for large dimensions N directly in RAM
void generateRandomMatrix(vector<vector<double>>& mat, int n) {
    mt19937 gen(42); // fixed seed for reproducibility
    uniform_real_distribution<double> dis(-10.0, 10.0);
    mat.assign(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = dis(gen);
        }
    }
}

int main(int argc, char* argv[]) {
    // Arguments: [fileA/sizeN] [fileB/threads] [fileC] [num_threads]
    int N = 0;
    int num_threads = 1;
    bool in_memory_mode = false;

    string arg1 = (argc > 1) ? argv[1] : "A.txt";
    
    // Check if mode is directly passing dimension size N (e.g. for N >= 1024)
    if (argc > 1 && isdigit(arg1[0])) {
        N = stoi(arg1);
        num_threads = (argc > 2) ? stoi(argv[2]) : 1;
        in_memory_mode = true;
    }

    vector<vector<double>> A, B, C;

    if (in_memory_mode) {
        generateRandomMatrix(A, N);
        generateRandomMatrix(B, N);
        C.assign(N, vector<double>(N, 0.0));
    } else {
        string fileA = arg1;
        string fileB = (argc > 2) ? argv[2] : "B.txt";
        string fileC = (argc > 3) ? argv[3] : "C.txt";
        if (argc > 4) num_threads = stoi(argv[4]);

        int nA = 0, nB = 0;
        if (!readMatrix(fileA, A, nA) || !readMatrix(fileB, B, nB) || nA != nB) {
            cerr << "Error loading matrix files or size mismatch!" << endl;
            return 1;
        }
        N = nA;
        C.assign(N, vector<double>(N, 0.0));
    }

    // Set OpenMP threads
    omp_set_num_threads(num_threads);

    // Measure execution time
    auto start_time = chrono::high_resolution_clock::now();

    // Parallelized Cache-optimized i-k-j Matrix Multiplication using OpenMP
    #pragma omp parallel for collapse(1) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end_time - start_time;

    double total_flops = 2.0 * N * N * N - N * N;

    if (!in_memory_mode && argc > 3) {
        writeMatrix(argv[3], C, N);
    }

    cout << "\n--- C++ OPENMP EXECUTION RESULTS ---" << endl;
    cout << "Matrix Dimension (N): " << N << "x" << N << endl;
    cout << "Threads Used: " << num_threads << endl;
    cout << "Workload (FLOP): " << fixed << setprecision(0) << total_flops << endl;
    cout << "Execution Time: " << fixed << setprecision(3) << elapsed.count() << " ms" << endl;

    return 0;
}