import os
import subprocess
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CPP_SOURCE = "matrix_multiply.cpp"
CPP_EXE = "./matrix_multiply" if os.name != "nt" else "matrix_multiply.exe"
RESULTS_DIR = "benchmark_results"
CSV_FILE = os.path.join(RESULTS_DIR, "openmp_results.csv")

# Updated sizes up to 4096
DEFAULT_SIZES = [128, 256, 512, 1024, 2048, 4096]

def compile_cpp():
    """Compile C++ code with OpenMP support and -O3 optimization."""
    print("Compiling C++ program with OpenMP (-fopenmp -O3)...")
    cmd = ["g++", "-std=c++17", "-O3", "-fopenmp", CPP_SOURCE, "-o", "matrix_multiply"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Compilation error:")
        print(res.stderr)
        return False
    print("Compilation successful!\n")
    return True

def read_matrix_file(filename):
    """Read resultant matrix from output text file."""
    with open(filename, "r") as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    data = [[float(x) for x in line.split()] for line in lines[1:] if line.strip()]
    return n, np.array(data)

def parse_cpp_time(stdout_text):
    """Parse execution time printed by C++ binary."""
    for line in stdout_text.splitlines():
        if "Execution Time:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                return float(parts[1].replace("ms", "").strip())
    return None

def verify_multiplication(size, file_a, file_b, file_c):
    """Verify matrix multiplication correctness using NumPy BLAS."""
    try:
        _, A = read_matrix_file(file_a)
        _, B = read_matrix_file(file_b)
        _, C_actual = read_matrix_file(file_c)

        # Compute reference result using NumPy BLAS
        C_expected = np.matmul(A, B)

        # Verify numerical accuracy with floating-point tolerance
        is_correct = np.allclose(C_actual, C_expected, rtol=1e-3, atol=1e-3)
        max_diff = np.max(np.abs(C_actual - C_expected))
        return is_correct, max_diff
    except Exception as e:
        print(f"Verification error: {e}")
        return False, -1.0

def save_results_to_csv(new_results):
    """Append benchmark data for new thread runs into CSV file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_new = pd.DataFrame(new_results)

    if os.path.exists(CSV_FILE):
        df_existing = pd.read_csv(CSV_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=["N", "Threads"], keep="last", inplace=True)
    else:
        df_combined = df_new

    df_combined.to_csv(CSV_FILE, index=False)
    print(f"\n[+] Results appended and saved to CSV: '{CSV_FILE}'")
    return df_combined

def plot_openmp_benchmarks(df):
    """Plot execution time and speedup curves across different thread counts."""
    plt.figure(figsize=(14, 6))

    # Plot 1: Execution Time vs Matrix Size (N)
    plt.subplot(1, 2, 1)
    threads_list = sorted(df["Threads"].unique())
    for t in threads_list:
        sub = df[df["Threads"] == t].sort_values("N")
        plt.plot(sub["N"], sub["CppTime_ms"], marker='o', linewidth=2, label=f'{t} Threads')
    
    plt.title('Execution Time vs Matrix Size (N)')
    plt.xlabel('Matrix Dimension (N)')
    plt.ylabel('Execution Time (ms)')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend()

    # Plot 2: Speedup vs Number of Threads
    plt.subplot(1, 2, 2)
    if 1 in threads_list:
        max_N = df["N"].max()
        sub_N = df[df["N"] == max_N].sort_values("Threads")
        t1_time = sub_N[sub_N["Threads"] == 1]["CppTime_ms"].values
        if len(t1_time) > 0:
            speedup = t1_time[0] / sub_N["CppTime_ms"]
            plt.plot(sub_N["Threads"], speedup, marker='s', color='crimson', linewidth=2, label=f'Speedup (N={max_N})')
            plt.plot(sub_N["Threads"], sub_N["Threads"], linestyle=':', color='gray', label='Ideal Speedup')
            plt.title(f'OpenMP Speedup for Matrix Size N={max_N}')
            plt.xlabel('Threads')
            plt.ylabel('Speedup Factor (T1 / T_N)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "openmp_benchmark_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[+] Performance plot saved to '{plot_path}'")
    plt.show()

def run_openmp_tests(threads, sizes=DEFAULT_SIZES):
    new_results = []
    print("=" * 70)
    print(f"STARTING OPENMP BENCHMARK (THREADS = {threads})")
    print("=" * 70)

    for size in sizes:
        print(f"\n---> Testing Dimension N = {size}x{size} on {threads} threads...")
        
        test_dir = os.path.join(RESULTS_DIR, f"run_N{size}")
        os.makedirs(test_dir, exist_ok=True)
        file_a = os.path.join(test_dir, "A.txt")
        file_b = os.path.join(test_dir, "B.txt")
        file_c = os.path.join(test_dir, "C.txt")

        # Generate and save input matrices A and B if they don't exist yet
        if not os.path.exists(file_a) or not os.path.exists(file_b):
            print("     Generating matrix input files A.txt and B.txt...")
            A = np.random.uniform(-10.0, 10.0, size=(size, size))
            B = np.random.uniform(-10.0, 10.0, size=(size, size))

            with open(file_a, "w") as f:
                f.write(f"{size}\n")
                np.savetxt(f, A, fmt="%.6f")

            with open(file_b, "w") as f:
                f.write(f"{size}\n")
                np.savetxt(f, B, fmt="%.6f")

        # Execute C++ executable reading A and B, and writing result to C
        cmd = [CPP_EXE, file_a, file_b, file_c, str(threads)]
        cpp_run = subprocess.run(cmd, capture_output=True, text=True)
        cpp_time = parse_cpp_time(cpp_run.stdout)

        # Automated verification via NumPy for all matrix sizes
        is_verified = False
        max_diff = -1.0
        if cpp_run.returncode == 0:
            is_verified, max_diff = verify_multiplication(size, file_a, file_b, file_c)
            status_str = "PASSED" if is_verified else "FAILED"
            print(f"     [NumPy Verification: {status_str}] (Max Diff: {max_diff:.8f})")

        if cpp_run.returncode != 0 or cpp_time is None:
            print(f"Error executing C++ program: {cpp_run.stderr}")
            continue

        flops = 2.0 * (size ** 3) - (size ** 2)
        print(f"Completed N={size:<4} | C++ Time: {cpp_time:>8.2f} ms | FLOP: {flops:.2e}")

        new_results.append({
            "N": size,
            "Threads": threads,
            "FLOP": flops,
            "CppTime_ms": cpp_time,
            "Verified": is_verified
        })

    df_all = save_results_to_csv(new_results)
    plot_openmp_benchmarks(df_all)

if __name__ == "__main__":
    if not compile_cpp():
        exit(1)

    max_cpus = os.cpu_count() or 4
    try:
        threads_input = input(f"Enter number of OpenMP threads/cores (available: {max_cpus}, default: {max_cpus}): ").strip()
        num_threads = int(threads_input) if threads_input else max_cpus
    except ValueError:
        print(f"Invalid input. Defaulting to: {max_cpus}")
        num_threads = max_cpus

    run_openmp_tests(num_threads)