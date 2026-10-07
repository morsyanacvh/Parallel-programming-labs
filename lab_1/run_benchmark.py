import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

CPP_SOURCE = "matrix_multiply.cpp"
CPP_EXE = "./matrix_multiply" if os.name != "nt" else "matrix_multiply.exe"
RESULTS_DIR = "benchmark_results"

def compile_cpp():
    """Automatically compile C++ code using C++17 standard with -O3 optimization."""
    print("Compiling C++ program...")
    cmd = ["g++", "-std=c++17", "-O3", CPP_SOURCE, "-o", "matrix_multiply"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Compilation error:")
        print(res.stderr)
        return False
    print("Compilation successful!\n")
    return True

def generate_matrix_files(size, file_a, file_b):
    """Generate input files with random matrices of a given size inside specified paths."""
    A = np.random.uniform(-10.0, 10.0, size=(size, size))
    B = np.random.uniform(-10.0, 10.0, size=(size, size))

    def save(filepath, mat):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(f"{size}\n")
            np.savetxt(f, mat, fmt="%.6f")

    save(file_a, A)
    save(file_b, B)
    return A, B

def read_matrix_file(filename):
    """Read resultant matrix from output file."""
    with open(filename, "r") as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    data = [[float(x) for x in line.split()] for line in lines[1:] if line.strip()]
    return n, np.array(data)

def parse_cpp_time(stdout_text):
    """Parse execution time printed by the C++ binary."""
    for line in stdout_text.splitlines():
        if "Execution Time:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                return float(parts[1].replace("ms", "").strip())
    return None

def plot_results(results):
    """Generate and save benchmark performance plots inside the results folder."""
    sizes = [r["N"] for r in results]
    flops = [r["FLOP"] for r in results]
    cpp_times = [r["CppTime_ms"] for r in results]
    py_times = [r["PyTime_ms"] for r in results]

    plt.figure(figsize=(12, 5))

    # Plot 1: Execution Time vs Matrix Size (N)
    plt.subplot(1, 2, 1)
    plt.plot(sizes, cpp_times, marker='o', linewidth=2, label='C++ (i-k-j loop)')
    plt.plot(sizes, py_times, marker='s', linewidth=2, label='Python (NumPy / BLAS)')
    plt.title('Execution Time vs Matrix Size (N)')
    plt.xlabel('Matrix Dimension (N)')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Plot 2: Execution Time vs Computational Workload (FLOP)
    plt.subplot(1, 2, 2)
    plt.plot(flops, cpp_times, marker='o', color='crimson', linewidth=2, label='C++')
    plt.title('Execution Time vs Task Workload (FLOP)')
    plt.xlabel('Task Workload (FLOP Count = 2N³ - N²)')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plot_filename = os.path.join(RESULTS_DIR, "benchmark_plot.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"\nPlot successfully generated and saved to '{plot_filename}'")
    plt.show()

def run_tests(sizes):
    results = []

    # Ensure main results folder exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for size in sizes:
        print("=" * 60)
        print(f"BENCHMARK FOR MATRIX DIMENSION N = {size}")
        print("=" * 60)

        # Create subfolder for specific test dimension
        test_dir = os.path.join(RESULTS_DIR, f"run_N{size}")
        os.makedirs(test_dir, exist_ok=True)

        file_a = os.path.join(test_dir, "A.txt")
        file_b = os.path.join(test_dir, "B.txt")
        file_c = os.path.join(test_dir, "C.txt")

        # 1. Generate matrices A and B inside test folder
        print(f"1. Generating random matrices ({size}x{size}) in '{test_dir}'...")
        A, B = generate_matrix_files(size, file_a, file_b)

        # 2. Compute reference result using NumPy
        t0 = time.perf_counter()
        C_expected = np.matmul(A, B)
        py_time_ms = (time.perf_counter() - t0) * 1000

        # 3. Execute C++ program with file path arguments
        print("2. Running C++ executable...")
        cpp_run = subprocess.run([CPP_EXE, file_a, file_b, file_c], capture_output=True, text=True)
        print(cpp_run.stdout)

        if cpp_run.returncode != 0:
            print(f"C++ execution failed: {cpp_run.stderr}")
            continue

        cpp_time_ms = parse_cpp_time(cpp_run.stdout)

        # 4. Verify results using NumPy
        _, C_actual = read_matrix_file(file_c)
        is_correct = np.allclose(C_actual, C_expected, rtol=1e-3, atol=1e-3)
        max_diff = np.max(np.abs(C_actual - C_expected))

        status = "PASSED" if is_correct else "FAILED"
        print(f"3. NumPy Verification: [{status}] (Max Absolute Difference: {max_diff:.8f})")

        flops = 2.0 * (size ** 3) - (size ** 2)
        results.append({
            "N": size,
            "FLOP": flops,
            "Passed": is_correct,
            "MaxDiff": max_diff,
            "CppTime_ms": cpp_time_ms if cpp_time_ms is not None else 0.0,
            "PyTime_ms": py_time_ms
        })

    # Print summary table
    print("\n" + "=" * 75)
    print(" BENCHMARK SUMMARY REPORT ")
    print("=" * 75)
    print(f"{'N (Size)':<10} | {'FLOP':<14} | {'C++ Time (ms)':<14} | {'NumPy Time (ms)':<16} | {'Status':<8}")
    print("-" * 75)
    for r in results:
        status_str = "OK" if r["Passed"] else "FAIL"
        print(f"{r['N']:<10} | {r['FLOP']:<14.0f} | {r['CppTime_ms']:<14.2f} | {r['PyTime_ms']:<16.2f} | {status_str:<8}")
    print("=" * 75)

    # Generate plots
    if results:
        plot_results(results)

if __name__ == "__main__":
    if not compile_cpp():
        exit(1)

    # Prompt user for matrix sizes
    user_input = input("Enter matrix dimensions separated by space or comma (e.g., 10, 100, 200, 400): ")
    try:
        sizes_list = [int(x.strip()) for x in user_input.replace(",", " ").split() if x.strip()]
        if not sizes_list:
            sizes_list = [10, 50, 100, 200, 300]
    except ValueError:
        print("Invalid input format. Falling back to default dimensions: [10, 50, 100, 200, 300]")
        sizes_list = [10, 50, 100, 200, 300]

    run_tests(sizes_list)