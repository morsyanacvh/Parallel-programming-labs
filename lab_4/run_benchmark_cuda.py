import os
import subprocess
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

CUDA_SOURCE = "matrix_multiply.cu"
CUDA_EXE = "./matrix_multiply_cuda" if os.name != "nt" else "matrix_multiply_cuda.exe"
RESULTS_DIR = "benchmark_results_cuda"


def compile_cuda():
    """Automatically compile CUDA code using nvcc with C++17 standard and -O3 optimization."""
    print("Compiling CUDA program with nvcc...")
    cmd = ["nvcc", "-std=c++17", "-O3", CUDA_SOURCE, "-o", "matrix_multiply_cuda"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Compilation error:")
        print(res.stderr)
        return False
    print("Compilation successful!\n")
    return True


def generate_matrix_files(size, file_a, file_b):
    """Generate input files with random matrices of a given size."""
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


def parse_cuda_time(stdout_text):
    """Parse kernel execution time printed by the CUDA binary."""
    for line in stdout_text.splitlines():
        if "Execution Time:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                return float(parts[1].replace("ms", "").strip())
    return None


def save_csv(results):
    """Save benchmark results to CSV."""
    csv_path = os.path.join(RESULTS_DIR, "cuda_benchmark_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["N", "BlockSize", "ThreadsPerBlock", "FLOP", "CudaTime_ms", "PyTime_ms", "MaxDiff", "Passed"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV report saved to '{csv_path}'")


def plot_results(results, sizes, block_sizes):
    """Generate and save benchmark performance plots for different grid/block sizes."""
    plt.figure(figsize=(13, 5.5))

    # Plot 1: Execution Time vs Matrix Size (N) for each CUDA Block Size
    plt.subplot(1, 2, 1)
    for bs in block_sizes:
        subset = [r for r in results if r["BlockSize"] == bs]
        subset.sort(key=lambda x: x["N"])
        x_vals = [r["N"] for r in subset]
        y_vals = [r["CudaTime_ms"] for r in subset]
        plt.plot(x_vals, y_vals, marker='o', linewidth=2, label=f'CUDA Block {bs}x{bs} ({bs*bs} thr)')

    # Also plot NumPy reference time
    py_subset = [r for r in results if r["BlockSize"] == block_sizes[0]]
    py_subset.sort(key=lambda x: x["N"])
    plt.plot(
        [r["N"] for r in py_subset],
        [r["PyTime_ms"] for r in py_subset],
        marker='s',
        linestyle='--',
        color='black',
        linewidth=1.5,
        label='CPU NumPy (BLAS)'
    )

    plt.title('CUDA Execution Time vs Matrix Size (N)')
    plt.xlabel('Matrix Dimension (N)')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Plot 2: Execution Time vs CUDA Block Size for each Matrix Dimension
    plt.subplot(1, 2, 2)
    block_labels = [f"{bs}x{bs}\n({bs*bs})" for bs in block_sizes]
    x_indices = np.arange(len(block_sizes))

    for size in sizes:
        subset = [r for r in results if r["N"] == size]
        subset.sort(key=lambda x: x["BlockSize"])
        y_vals = [r["CudaTime_ms"] for r in subset]
        plt.plot(x_indices, y_vals, marker='o', linewidth=2, label=f'N = {size}')

    plt.xticks(x_indices, block_labels)
    plt.title('Execution Time vs CUDA Block/Grid Configuration')
    plt.xlabel('Block Dimensions (Threads per Block)')
    plt.ylabel('Kernel Execution Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plot_filename = os.path.join(RESULTS_DIR, "cuda_benchmark_plot.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot successfully generated and saved to '{plot_filename}'")
    plt.show()


def run_tests(sizes, block_sizes):
    results = []
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for size in sizes:
        print("=" * 70)
        print(f"BENCHMARK FOR MATRIX DIMENSION N = {size}")
        print("=" * 70)

        test_dir = os.path.join(RESULTS_DIR, f"run_N{size}")
        os.makedirs(test_dir, exist_ok=True)

        file_a = os.path.join(test_dir, "A.txt")
        file_b = os.path.join(test_dir, "B.txt")

        # 1. Generate matrices once per dimension N
        print(f"1. Generating random matrices ({size}x{size}) in '{test_dir}'...")
        A, B = generate_matrix_files(size, file_a, file_b)

        # 2. Compute reference result using NumPy
        t0 = time.perf_counter()
        C_expected = np.matmul(A, B)
        py_time_ms = (time.perf_counter() - t0) * 1000
        flops = 2.0 * (size ** 3) - (size ** 2)

        # 3. Test across all requested CUDA block sizes (grid configurations)
        for bs in block_sizes:
            file_c = os.path.join(test_dir, f"C_block{bs}.txt")
            print(f"\n2. Running CUDA executable with block size {bs}x{bs} ({bs*bs} threads/block)...")
            cuda_run = subprocess.run(
                [CUDA_EXE, file_a, file_b, file_c, str(bs)],
                capture_output=True,
                text=True
            )
            print(cuda_run.stdout.strip())

            if cuda_run.returncode != 0:
                print(f"CUDA execution failed: {cuda_run.stderr}")
                continue

            cuda_time_ms = parse_cuda_time(cuda_run.stdout)

            # 4. Verify results using NumPy
            _, C_actual = read_matrix_file(file_c)
            is_correct = np.allclose(C_actual, C_expected, rtol=1e-3, atol=1e-3)
            max_diff = float(np.max(np.abs(C_actual - C_expected)))

            status = "PASSED" if is_correct else "FAILED"
            print(f"3. NumPy Verification [Block {bs}x{bs}]: [{status}] (Max Diff: {max_diff:.8f})")

            results.append({
                "N": size,
                "BlockSize": bs,
                "ThreadsPerBlock": bs * bs,
                "FLOP": flops,
                "CudaTime_ms": cuda_time_ms if cuda_time_ms is not None else 0.0,
                "PyTime_ms": py_time_ms,
                "MaxDiff": max_diff,
                "Passed": is_correct
            })

    # Print summary table
    print("\n" + "=" * 95)
    print(" CUDA BENCHMARK SUMMARY REPORT ")
    print("=" * 95)
    print(f"{'N (Size)':<10} | {'Block':<9} | {'Grid':<11} | {'CUDA Time (ms)':<15} | {'NumPy Time (ms)':<16} | {'Status':<8}")
    print("-" * 95)
    for r in results:
        bs = r["BlockSize"]
        grid_dim = (r["N"] + bs - 1) // bs
        block_str = f"{bs}x{bs}"
        grid_str = f"{grid_dim}x{grid_dim}"
        status_str = "OK" if r["Passed"] else "FAIL"
        print(
            f"{r['N']:<10} | {block_str:<9} | {grid_str:<11} | "
            f"{r['CudaTime_ms']:<15.3f} | {r['PyTime_ms']:<16.3f} | {status_str:<8}"
        )
    print("=" * 95)

    if results:
        save_csv(results)
        plot_results(results, sizes, block_sizes)


if __name__ == "__main__":
    if not compile_cuda():
        exit(1)

    # Prompt user for matrix sizes
    sizes_input = input("Enter matrix dimensions separated by space/comma (default: 256, 512, 1024, 2048): ")
    try:
        sizes_list = sorted(list(set(int(x.strip()) for x in sizes_input.replace(",", " ").split() if x.strip())))
        if not sizes_list:
            sizes_list = [256, 512, 1024, 2048]
    except ValueError:
        print("Invalid input. Using default dimensions: [256, 512, 1024, 2048]")
        sizes_list = [256, 512, 1024, 2048]

    # Prompt user for CUDA block sizes (which automatically determines the grid size: ceil(N / block_size))
    blocks_input = input("Enter CUDA block sizes (1-32) separated by space/comma (default: 8, 16, 32): ")
    try:
        blocks_list = sorted(list(set(
            int(x.strip()) for x in blocks_input.replace(",", " ").split()
            if x.strip() and 1 <= int(x.strip()) <= 32
        )))
        if not blocks_list:
            blocks_list = [8, 16, 32]
    except ValueError:
        print("Invalid input. Using default block sizes: [8, 16, 32]")
        blocks_list = [8, 16, 32]

    run_tests(sizes_list, blocks_list)
