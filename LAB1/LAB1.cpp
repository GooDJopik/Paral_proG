#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;

// Функция для чтения матрицы из файла
vector<vector<int>> readMatrix(const string& filename, int& rows, int& cols) {
    ifstream file(filename);
    file >> rows >> cols;
    vector<vector<int>> matrix(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            file >> matrix[i][j];
    return matrix;
}

// Функция для записи матрицы в файл
void writeMatrix(const string& filename, const vector<vector<int>>& matrix, double execution_time, int operation_count) {
    ofstream file(filename);
    int rows = matrix.size();
    int cols = matrix[0].size();
    file << rows << " " << cols << endl;
    for (const auto& row : matrix) {
        for (int value : row)
            file << value << " ";
        file << endl;
    }
    // Запись времени выполнения и объема задачи
    file << "Lead time: " << execution_time << " seconds" << endl;
    file << "Task scope: " << operation_count << " operations" << endl;
}

// Функция для перемножения матриц
vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();
    vector<vector<int>> C(rowsA, vector<int>(colsB, 0));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

int main() {
    int rowsA, colsA, rowsB, colsB;

    // Чтение матриц из файлов
    auto A = readMatrix("matrixA.txt", rowsA, colsA);
    auto B = readMatrix("matrixB.txt", rowsB, colsB);

    // Проверка возможности перемножения
    if (colsA != rowsB) {
        cerr << "Error: the number of columns of the first matrix does not match the number of rows of the second matrix." << endl;
        return 1;
    }

    // Измерение времени выполнения
    auto start = chrono::high_resolution_clock::now();
    auto C = multiplyMatrices(A, B);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    // Вычисление объема задачи
    int operation_count = rowsA * colsA * colsB;

    // Запись результата в файл
    writeMatrix("result.txt", C, duration.count(), operation_count);

    // Вывод времени выполнения и объема задачи
    cout << "Lead time: " << duration.count() << " seconds" << endl;
    cout << "Task scope: " << operation_count << " operations" << endl;

    return 0;
}
