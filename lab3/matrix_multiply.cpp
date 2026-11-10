#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <string>
#include <random>
#include <mpi.h>

using namespace std;

// Function to read a square matrix from a file
bool readMatrix(const string& filepath, vector<vector<double>>& matrix, int& n) {
    ifstream file(filepath.c_str());
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

// Function to write a square matrix to a file
bool writeMatrix(const string& filepath, const vector<vector<double>>& matrix, int n) {
    ofstream file(filepath.c_str());
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

void generateRandomMatrix(vector<vector<double>>& mat, int n) {
    mt19937 gen(42);
    uniform_real_distribution<double> dis(-10.0, 10.0);
    mat.assign(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = dis(gen);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0;
    bool in_memory_mode = false;
    string fileA, fileB, fileC;

    if (argc > 2) {
        fileA = argv[1];
        fileB = argv[2];
        fileC = argv[3];
        in_memory_mode = false;
    } else if (argc == 2) {
        N = stoi(argv[1]);
        in_memory_mode = true;
    } else {
        if (rank == 0) cerr << "Usage: mpirun -np <procs> ./matrix_multiply <fileA> <fileB> <fileC> OR <N>" << endl;
        MPI_Finalize();
        return 1;
    }

    vector<vector<double>> A, B, C;

    if (rank == 0) {
        if (in_memory_mode) {
            generateRandomMatrix(A, N);
            generateRandomMatrix(B, N);
        } else {
            int nA = 0, nB = 0;
            if (!readMatrix(fileA, A, nA) || !readMatrix(fileB, B, nB) || nA != nB) {
                cerr << "Error loading matrix files!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            N = nA;
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        B.assign(N, vector<double>(N));
    }
    vector<double> flatB(N * N);
    if (rank == 0) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                flatB[i * N + j] = B[i][j];
    }
    MPI_Bcast(flatB.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                B[i][j] = flatB[i * N + j];
    }

    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    vector<vector<double>> local_A(local_rows, vector<double>(N));
    vector<vector<double>> local_C(local_rows, vector<double>(N, 0.0));

    vector<int> sendcounts(size);
    vector<int> displs(size);
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        int r = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = r * N;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    vector<double> flatA;
    if (rank == 0) {
        flatA.resize(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                flatA[i * N + j] = A[i][j];
    }

    vector<double> local_flatA(local_rows * N);
    MPI_Scatterv(rank == 0 ? flatA.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_flatA.data(), local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            local_A[i][j] = local_flatA[i * N + j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = chrono::high_resolution_clock::now();

    for (int i = 0; i < local_rows; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end_time - start_time;

    vector<double> local_flatC(local_rows * N);
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < N; ++j)
            local_flatC[i * N + j] = local_C[i][j];

    vector<double> flatC;
    if (rank == 0) {
        flatC.resize(N * N);
    }

    MPI_Gatherv(local_flatC.data(), local_rows * N, MPI_DOUBLE,
                rank == 0 ? flatC.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        C.assign(N, vector<double>(N));
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                C[i][j] = flatC[i * N + j];

        if (!in_memory_mode) {
            writeMatrix(fileC, C, N);
        }

        double total_flops = 2.0 * N * N * N - N * N;
        cout << "\n--- C++ MPI EXECUTION RESULTS ---" << endl;
        cout << "Matrix Dimension (N): " << N << "x" << N << endl;
        cout << "MPI Processes Used: " << size << endl;
        cout << "Workload (FLOP): " << fixed << setprecision(0) << total_flops << endl;
        cout << "Execution Time: " << fixed << setprecision(3) << elapsed.count() << " ms" << endl;
    }

    MPI_Finalize();
    return 0;
}