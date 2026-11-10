#!/bin/bash
#SBATCH --job-name=matrix_full_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --output=full_benchmark.log
#SBATCH --error=full_benchmark_error.log

# Переходим в рабочую директорию
cd $SLURM_SUBMIT_DIR

# Сбрасываем старые модули и загружаем OpenMPI
module purge
module load intel/mpi5

# Списки тестируемых параметров
MATRIX_SIZES=(128 256 512 1024 2048 4096)
PROCESS_COUNTS=(1 2 4 8 16)

# Создаем файл результатов с расширенной структурой (добавили колонку Size)
echo "Processes,Size,Time_ms" > benchmark_results.csv

echo "=================================================="
echo " Starting Full Matrix Benchmark Suite"
echo "=================================================="

for size in "${MATRIX_SIZES[@]}"; do
    for np in "${PROCESS_COUNTS[@]}"; do
        
        # Пропускаем случаи, если процессов больше, чем размер матрицы (не имеет смысла)
        if [ $np -gt $size ]; then
            continue
        fi

        echo "--------------------------------------------------"
        echo "-> Testing Matrix Size: ${size}x${size} | Processes (np): $np"
        
        # Запускаем программу с конкретным числом процессов и текущим размером матрицы
        output=$(srun -n $np ./matrix_multiply $size)
        
        # Извлекаем время выполнения через надежное регулярное выражение
        time_ms=$(echo "$output" | grep "Execution Time" | sed -E 's/.*Execution Time: *([0-9.]+).*/\1/')
        
        if [ ! -z "$time_ms" ]; then
            # Записываем в формате: Processes,Size,Time_ms
            echo "$np,$size,$time_ms" >> benchmark_results.csv
            echo "[SUCCESS] Size=$size, np=$np -> $time_ms ms"
        else
            echo "[WARNING] Could not parse time for Size=$size, np=$np"
        fi
        
    done
done

echo "=================================================="
echo " Benchmark Suite Completed Successfully!"
echo " Results saved to benchmark_results.csv"
echo "=================================================="