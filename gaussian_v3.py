import numpy as np
import multiprocessing as mp

def eliminate_row(shared_Ab, i, j, n, lock):
    """
    Функция для выполнения элементарного преобразования строк в прямом ходе метода Гаусса.
    Используется мьютекс для блокировки доступа к разделяемой памяти shared_Ab.
    """
    with lock:
        # Загружаем разделяемый массив в виде numpy матрицы
        Ab = np.frombuffer(shared_Ab.get_obj()).reshape(n, n + 1)
        # Вычисляем коэффициент для вычитания строки i из строки j
        factor = Ab[j, i] / Ab[i, i]
        # Проводим преобразование строки
        Ab[j, i:] -= factor * Ab[i, i:]

def gaussian_elimination_parallel(A, b):
    """
    Основная функция для решения СЛАУ методом Гаусса с использованием параллелизма.
    """
    n = len(b)
    # Объединяем матрицу A и вектор b в расширенную матрицу
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    # Создаем общий массив для параллельного доступа, преобразуем его в одномерный массив
    shared_Ab = mp.Array('d', Ab.flatten())
    
    # Создаем мьютекс для управления доступом к разделяемым данным
    lock = mp.Lock()

    # Прямой ход метода Гаусса с использованием параллельных процессов
    for i in range(n):
        # Преобразуем shared_Ab обратно в numpy для текущего процесса
        Ab = np.frombuffer(shared_Ab.get_obj()).reshape(n, n + 1)

        # Находим индекс строки с максимальным элементом для обмена (частичная выборка главного элемента)
        max_row_index = np.argmax(np.abs(Ab[i:, i])) + i
        # Меняем строки для улучшения численной устойчивости
        if i != max_row_index:
            Ab[[i, max_row_index]] = Ab[[max_row_index, i]]
        
        # Параллельно выполняем преобразования для строк ниже i, создавая процессы вручную
        processes = []
        for j in range(i + 1, n):
            p = mp.Process(target=eliminate_row, args=(shared_Ab, i, j, n, lock))
            processes.append(p)
            p.start()

        # Ждем завершения всех процессов
        for p in processes:
            p.join()

    # Обратный ход для получения значений переменных
    x = np.zeros(n)
    Ab = np.frombuffer(shared_Ab.get_obj()).reshape(n, n + 1)  # Последнее преобразование в numpy

    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1]  # Заполняем x начиная с конца
        for j in range(i + 1, n):
            x[i] -= Ab[i, j] * x[j]
        x[i] /= Ab[i, i]

    return x

# Функция для чтения матрицы и вектора из файла
def read_matrix_and_vector(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Разделение матрицы и вектора
    A = []
    for line in lines[:-1]:  # Все строки, кроме последней, — это матрица A
        A.append(list(map(float, line.split())))

    b = list(map(float, lines[-1].split()))  # Последняя строка — это вектор b
    
    return np.array(A), np.array(b)

# Пример использования функции
if __name__ == "__main__":
    # Чтение матрицы и вектора из файла
    A, b = read_matrix_and_vector('matrix.txt')

    # Запуск решения
    solution = gaussian_elimination_parallel(A, b)
    print("Решение СЛАУ:", np.round(solution, decimals=6))
