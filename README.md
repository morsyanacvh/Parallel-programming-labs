# Matrix Multiplication Benchmark & Verification (C++ / Python)

This project performs square matrix multiplication in C++ with optimal memory layout ($O(N^3)$ algorithm), measures runtime and FLOP workload, and automatically verifies numerical correctness using Python (NumPy).

## Features
- **C++ Core**: High-performance $O(N^3)$ matrix multiplication.
- **Python Benchmark Automation**: Runs tests across multiple matrix dimensions, checks precision (`numpy.allclose`), and logs execution metrics.
- **Visualization & Reporting**: Saves performance plots and exports detailed Markdown reports.

## Requirements
- `g++` compiler supporting C++17
- Python 3.x
- Python packages: `numpy`, `matplotlib`

```bash
pip install numpy matplotlib