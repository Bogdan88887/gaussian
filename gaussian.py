import numpy as np
from multiprocessing import Pool, cpu_count

# Функция для параллельного вычитания строк
def eliminate_row(args):
    Ab, i, k = args
    factor = Ab[k, i] / Ab[i, i]
    Ab[k, i:] -= factor * Ab[i, i:]

# Прямой ход метода Гаусса
def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])  # расширенная матрица

    for i in range(n):
        # Нахождение ведущего элемента для текущего столбца
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]  # Перестановка строк

        # Параллельное выполнение вычитания строк
        with Pool(cpu_count()) as p:
            p.map(eliminate_row, [(Ab, i, k) for k in range(i + 1, n)])

    # Обратный ход для нахождения решений
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    return x

# Пример использования
if __name__ == '__main__':
    # Пример матрицы A и вектора b
    A = np.array([[3.0, 2.0, -4.0],
                  [2.0, 3.0, 3.0],
                  [5.0, -3.0, 1.0]])

    b = np.array([3.0, 15.0, 14.0])

    # Решение СЛАУ методом Гаусса
    x = gaussian_elimination(A, b)
    print("Решение СЛАУ:", x)
