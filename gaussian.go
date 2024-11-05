package main

import (
 "bufio"
 "fmt"
 "math"
 "os"
 "strconv"
 "strings"
 "sync"
)

func eliminateRow(Ab [][]float64, i int, wg *sync.WaitGroup) {
 defer wg.Done()
 n := len(Ab)
 for j := i + 1; j < n; j++ {
  factor := Ab[j][i] / Ab[i][i]
  for k := i; k <= n; k++ {
   Ab[j][k] -= factor * Ab[i][k]
  }
 }
}

func gaussianElimination(A [][]float64, b []float64, result chan []float64) {
 n := len(b)
 // Создаем расширенную матрицу Ab
 Ab := make([][]float64, n)
 for i := range Ab {
  Ab[i] = append(A[i], b[i])
 }

 // Прямой ход
 for i := 0; i < n; i++ {
  maxRowIndex := i
  maxValue := math.Abs(Ab[i][i])
  for k := i + 1; k < n; k++ {
   if math.Abs(Ab[k][i]) > maxValue {
    maxRowIndex = k
    maxValue = math.Abs(Ab[k][i])
   }
  }
  Ab[i], Ab[maxRowIndex] = Ab[maxRowIndex], Ab[i]

  var wg sync.WaitGroup
  for j := i + 1; j < n; j++ {
   wg.Add(1)
   go eliminateRow(Ab, i, &wg)
  }
  wg.Wait()
 }

 x := make([]float64, n)
 for i := n - 1; i >= 0; i-- {
  x[i] = Ab[i][n]
  for j := i + 1; j < n; j++ {
   x[i] -= Ab[i][j] * x[j]
  }
  x[i] /= Ab[i][i]
 }
 result <- x
}

func readMatrixAndVectorFromFile(filename string) ([][]float64, []float64, error) {
 file, err := os.Open(filename)
 if err != nil {
  return nil, nil, err
 }
 defer file.Close()

 scanner := bufio.NewScanner(file)
 var A [][]float64
 var b []float64

 // Читаем строки из файла
 lines := []string{}
 for scanner.Scan() {
  lines = append(lines, scanner.Text())
 }

 if err := scanner.Err(); err != nil {
  return nil, nil, err
 }

 // Обрабатываем строки, формируя матрицу A и вектор b
 for i := 0; i < len(lines)-1; i++ {
  line := strings.Fields(lines[i])
  row := make([]float64, len(line))
  for j, value := range line {
   row[j], _ = strconv.ParseFloat(value, 64)
  }
  A = append(A, row)
 }

 // Обрабатываем последнюю строку как вектор b
 if len(lines) > 0 {
  line := strings.Fields(lines[len(lines)-1])
  b = make([]float64, len(line))
  for i, value := range line {
   b[i], _ = strconv.ParseFloat(value, 64)
  }
 }

 return A, b, nil
}

func main() {
 // Чтение матрицы и вектора из файла
 A, b, err := readMatrixAndVectorFromFile("matrix.txt")
 if err != nil {
  fmt.Println("Ошибка при чтении файла:", err)
  return
 }

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
