package main

import (
	"fmt"
	"math"
	"time"
)

func gaussianElimination(A [][]float64, b []float64, result chan []float64) {
	n := len(b)
	// Создаем расширенную матрицу Ab
	Ab := make([][]float64, n)
	for i := range Ab {
		Ab[i] = append(A[i], b[i])
	}

	// Прямой ход
	for i := 0; i < n; i++ {
		// Нормализуем строку с максимальным элементом
		maxRowIndex := i
		maxValue := math.Abs(Ab[i][i])
		for k := i + 1; k < n; k++ {
			if math.Abs(Ab[k][i]) > maxValue {
				maxRowIndex = k
				maxValue = math.Abs(Ab[k][i])
			}
		}

		// Переставляем строки
		Ab[i], Ab[maxRowIndex] = Ab[maxRowIndex], Ab[i]

		// Приведение к треугольной форме
		for j := i + 1; j < n; j++ {
			factor := Ab[j][i] / Ab[i][i]
			for k := i; k <= n; k++ {
				Ab[j][k] -= factor * Ab[i][k]
			}
		}

		// Асинхронная пауза (имитация)
		time.Sleep(10 * time.Millisecond)
	}

	// Обратный ход
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = Ab[i][n] // Последний элемент в строке
		for j := i + 1; j < n; j++ {
			x[i] -= Ab[i][j] * x[j]
		}
		x[i] /= Ab[i][i]

		// Асинхронная пауза (имитация)
		time.Sleep(10 * time.Millisecond)
	}

	// Отправляем результат через канал
	result <- x
}

func main() {
	A := [][]float64{
		{1.0, 2.0, 3.0},
		{0.0, 1.0, 4.0},
		{5.0, 6.0, 0.0},
	}

	b := []float64{14.0, 10.0, 32.0}

	// Канал для получения результата
	result := make(chan []float64)

	// Запуск асинхронной функции в гоурутине
	go gaussianElimination(A, b, result)

	// Получение решения из канала
	solution := <-result

	// Выводим результат
	fmt.Printf("Решение СЛАУ: ")
	for _, v := range solution {
		fmt.Printf("%.6f ", v)
	}
	fmt.Println()
}
