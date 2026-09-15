#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <string>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

// Function to read a square matrix from a file
bool readMatrix(const string& filepath, vector<vector<double>>& matrix, int& n) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filepath << endl;
        return false;
    }

    if (!(file >> n) || n <= 0) {
        cerr << "Invalid matrix size in file: " << filepath << endl;
        return false;
    }

    matrix.assign(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(file >> matrix[i][j])) {
                cerr << "Error reading matrix elements from " << filepath << endl;
                return false;
            }
        }
    }
    file.close();
    return true;
}

// Function to write a square matrix to a file
bool writeMatrix(const string& filepath, const vector<vector<double>>& matrix, int n) {
    // Ensure parent directory exists before writing
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
            file << matrix[i][j] << (j == n - 1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
    return true;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments or use default file paths
    string fileA = (argc > 1) ? argv[1] : "A.txt";
    string fileB = (argc > 2) ? argv[2] : "B.txt";
    string fileC = (argc > 3) ? argv[3] : "C.txt";

    vector<vector<double>> A, B;
    int nA = 0, nB = 0;

    if (!readMatrix(fileA, A, nA) || !readMatrix(fileB, B, nB)) {
        return 1;
    }

    if (nA != nB) {
        cerr << "Error: Matrix dimensions do not match (" << nA << " != " << nB << ")" << endl;
        return 1;
    }

    int N = nA;
    vector<vector<double>> C(N, vector<double>(N, 0.0));

    // Measure execution time
    auto start_time = chrono::high_resolution_clock::now();

    // Cache-optimized matrix multiplication (i-k-j loop order)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end_time - start_time;

    // Calculate computational workload (FLOP count)
    double total_flops = 2.0 * N * N * N - N * N;

    if (!writeMatrix(fileC, C, N)) {
        return 1;
    }

    cout << "\n--- C++ EXECUTION RESULTS ---" << endl;
    cout << "Matrix Dimension (N): " << N << "x" << N << endl;
    cout << "Total Elements: " << N * N << endl;
    cout << "Workload (FLOP): " << fixed << setprecision(0) << total_flops << endl;
    cout << "Execution Time: " << fixed << setprecision(3) << elapsed.count() << " ms" << endl;
    cout << "Output File Saved: " << fileC << endl;

    return 0;
}