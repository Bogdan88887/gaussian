import numpy as np
import multiprocessing as mp

def eliminate_row(Ab, i, j):
    """Функция для параллельного вычисления одной строки в прямом ходе"""
    factor = Ab[j, i] / Ab[i, i]
    Ab[j, i:] -= factor * Ab[i, i:]
    return j, Ab[j]  # Возвращаем индекс строки и её новое значение

def gaussian_elimination_parallel(A, b):
    n = len(b)
    # Объединяем матрицу A и вектор b в расширенную матрицу
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Прямой ход с использованием multiprocessing.Pool
    for i in range(n):
        # Нормализуем строку с максимальным элементом
        max_row_index = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row_index]] = Ab[[max_row_index, i]]
        
        # Создаем пул процессов
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Параллельное выполнение для строк ниже текущей строки i
            results = pool.starmap(eliminate_row, [(Ab, i, j) for j in range(i + 1, n)])
        
            # Обновляем строки матрицы с параллельных результатов
            for j, row in results:
                Ab[j] = row

    # Обратный ход без параллельных вычислений
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1]  # Начинаем с последнего столбца
        for j in range(i + 1, n):
            x[i] -= Ab[i, j] * x[j]
        x[i] /= Ab[i, i]

    return x

# Задаем матрицу A и вектор b
A = np.array([
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 4.0],
    [5.0, 6.0, 0.0]
])

b = np.array([14.0, 10.0, 32.0])

# Запускаем параллельное решение
solution = gaussian_elimination_parallel(A, b)
print("Решение СЛАУ:", np.round(solution, decimals=6))
