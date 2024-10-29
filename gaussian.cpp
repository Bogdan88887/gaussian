#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

using namespace std;

// Функция для выполнения метода Гаусса
vector<double> gaussianElimination(vector<vector<double>>& A, vector<double>& b) {
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

        // Приведение к треугольной форме
        for (int j = i + 1; j < n; ++j) {
            double factor = Ab[j][i] / Ab[i][i];
            for (int k = i; k <= n; ++k) {
                Ab[j][k] -= factor * Ab[i][k];
            }
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

int main() {
    // Задаем матрицу A и вектор b
    vector<vector<double>> A = {
        {4.0, -2.0, 1.0},
        {3.0, 6.0, -4.0},
        {2.0, 1.0, 8.0}
    };

    vector<double> b = {12.0, -25.0, 20.0};

    // Решение системы
    vector<double> solution = gaussianElimination(A, b);

    // Выводим результат
    cout << "Решение СЛАУ: ";
    for (double val : solution) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
