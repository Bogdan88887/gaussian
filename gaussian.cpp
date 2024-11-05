#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>
#include <fstream>  // Для работы с файлами

using namespace std;

mutex mtx;  // Мьютекс для синхронизации вывода и доступа к данным

// Функция для приведения строки к треугольной форме параллельно
void rowElimination(vector<vector<double>>& Ab, int i, int j, int n) {
    double factor = Ab[j][i] / Ab[i][i];
    for (int k = i; k <= n; ++k) {
        Ab[j][k] -= factor * Ab[i][k];
    }
}

// Функция для выполнения метода Гаусса с параллельной обработкой
vector<double> gaussianEliminationParallel(vector<vector<double>>& A, vector<double>& b) {
    int n = b.size();
    // Создаем расширенную матрицу Ab
    vector<vector<double>> Ab(n, vector<double>(n + 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Ab[i][j] = A[i][j];
        }
        Ab[i][n] = b[i];
    }

    // Прямой ход
    for (int i = 0; i < n; ++i) {
        // Нормализуем строку с максимальным элементом
        int maxRowIndex = i;
        double maxValue = abs(Ab[i][i]);
        for (int k = i + 1; k < n; ++k) {
            if (abs(Ab[k][i]) > maxValue) {
                maxRowIndex = k;
                maxValue = abs(Ab[k][i]);
            }
        }

        // Переставляем строки
        swap(Ab[i], Ab[maxRowIndex]);

        // Параллельное приведение к треугольной форме
        vector<thread> threads;
        for (int j = i + 1; j < n; ++j) {
            threads.emplace_back(rowElimination, ref(Ab), i, j, n);
        }

        // Ожидание завершения всех потоков
        for (auto& th : threads) {
            th.join();
        }

        // Имитация асинхронной паузы
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    // Обратный ход
    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = Ab[i][n];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= Ab[i][j] * x[j];
        }
        x[i] /= Ab[i][i];

        // Имитация асинхронной паузы
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    return x;
}

// Функция для чтения матрицы и вектора из файла
void readMatrixFromFile(const string& filename, vector<vector<double>>& A, vector<double>& b) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть файл " << filename << endl;
        return;
    }

    double value;
    vector<double> row;
    int rowCount = 0; // Счетчик строк матрицы A

    // Чтение матрицы A
    while (file >> value) {
        if (rowCount < 3) { // Предполагаем, что у нас 3 строки в A
            row.push_back(value);
            if (row.size() == 3) { // 3 значения для каждой строки
                A.push_back(row);
                row.clear();
                rowCount++;
            }
        } else {
            b.push_back(value); // Все оставшиеся значения идут в b
        }
    }

    file.close();
}

int main() {
    vector<vector<double>> A;  // Инициализируем пустую матрицу A
    vector<double> b;          // Инициализируем пустой вектор b

    // Чтение матрицы и вектора из файла
    readMatrixFromFile("matrix.txt", A, b);

    // Проверяем, что данные успешно считаны
    if (A.empty() || b.empty()) {
        cerr << "Ошибка: матрица или вектор b пусты." << endl;
        return 1;
    }

    // Решение системы
    vector<double> solution = gaussianEliminationParallel(A, b);

    // Выводим результат
    mtx.lock();  // Блокируем вывод для синхронизации
    cout << "Решение СЛАУ: ";
    for (double val : solution) {
        cout << val << " ";
    }
    cout << endl;
    mtx.unlock();  // Разблокируем вывод

    return 0;
}
