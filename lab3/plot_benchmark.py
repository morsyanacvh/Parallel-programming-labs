import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Загружаем данные из CSV-файла (предполагаем колонки: Processes, MatrixSize, Time_ms)
# Если у вас пока сохраняются только Processes и Time_ms, убедитесь, 
# что скрипт бенчмаркинга также сохраняет размер матрицы (Size).
try:
    df = pd.read_csv('benchmark_results.csv')
except FileNotFoundError:
    print("[-] Файл benchmark_results.csv не найден! Сначала запустите бенчмарк.")
    exit(1)

# Проверим структуру данных (ожидаем колонки: Processes, Size, Time_ms)
print("Загруженные данные:")
print(df.head())

# --- ГРАФИК 1: Зависимость времени от размера матрицы (при фиксированном числе потоков) ---
plt.figure(figsize=(10, 6))

for procs, group in df.groupby('Processes'):
    plt.plot(group['Size'], group['Time_ms'], marker='o', label=f'{procs} потоков(а)')

plt.title('Зависимость времени выполнения от размера матрицы', fontsize=14)
plt.xlabel('Размер матрицы (N)', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('time_vs_size.png', dpi=300)
print("[+] График 1 сохранен как time_vs_size.png")
plt.close()

# --- ГРАФИК 2: Зависимость ускорения от количества потоков (для каждого размера матрицы) ---
# Ускорение (Speedup) рассчитывается как: T(1_thread) / T(N_threads)
plt.figure(figsize=(10, 6))

for size, group in df.groupby('Size'):
    # Сортируем по числу процессов
    group = group.sort_values('Processes')
    
    # Находим базовое время на 1 процессе для данного размера матрицы
    base_time_row = group[group['Processes'] == 1]
    if base_time_row.empty:
        continue # Если нет теста для 1 потока, пропуск
    
    base_time = base_time_row['Time_ms'].values[0]
    
    # Считаем ускорение
    speedups = base_time / group['Time_ms']
    
    plt.plot(group['Processes'], speedups, marker='s', label=f'N = {size}')

# Добавим идеальное линейное ускорение (для наглядности)
max_procs = df['Processes'].max()
if max_procs > 1:
    ideal_x = [1, max_procs]
    ideal_y = [1, max_procs]
    plt.plot(ideal_x, ideal_y, 'k--', label='Идеальное ускорение', alpha=0.5)

plt.title('Зависимость ускорения от количества потоков (Speedup)', fontsize=14)
plt.xlabel('Количество вычислительных потоков (процессов)', fontsize=12)
plt.ylabel('Ускорение (Сэкономленный раз)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('speedup_vs_procs.png', dpi=300)
print("[+] График 2 сохранен как speedup_vs_procs.png")
plt.close()