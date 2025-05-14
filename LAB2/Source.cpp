#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <omp.h>

using namespace std;

vector<vector<int>> generateRandomMatrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 10);

#pragma omp parallel for
    for (int idx = 0; idx < rows * cols; ++idx) {
        int i = idx / cols;
        int j = idx % cols;
        matrix[i][j] = dis(gen);
    }

    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<int>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        return;
    }

    int rows = matrix.size();
    int cols = matrix[0].size();
    file << rows << " " << cols << endl;
    for (const auto& row : matrix) {
        for (int value : row)
            file << value << " ";
        file << endl;
    }
}

vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();
    vector<vector<int>> C(rowsA, vector<int>(colsB, 0));

#pragma omp parallel for
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            int sum = 0;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

void writeTimeResults(const string& filename, const vector<pair<string, double>>& results) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла результатов: " << filename << endl;
        return;
    }

    file << "Размер матриц\tВремя выполнения (секунд)\tУскорение" << endl;

    double base_time = results.empty() ? 1.0 : results[0].second;
    for (const auto& result : results) {
        double speedup = base_time / result.second;
        file << result.first << "\t" << fixed << setprecision(6)
            << result.second << "\t" << speedup << endl;
    }
}

int main() {

    setlocale(LC_ALL, "Russian");

    omp_set_num_threads(omp_get_max_threads());
    cout << "Используется потоков: " << omp_get_max_threads() << endl;

    vector<int> sizes = {100, 200, 300, 400, 500, 1000, 1500};
    vector<pair<string, double>> timeResults;

    system("mkdir -p results");

    for (int size : sizes) {
        auto start_gen = chrono::high_resolution_clock::now();
        auto A = generateRandomMatrix(size, size);
        auto B = generateRandomMatrix(size, size);
        auto end_gen = chrono::high_resolution_clock::now();
        chrono::duration<double> gen_duration = end_gen - start_gen;

        writeMatrix("results/matrixA_" + to_string(size) + ".txt", A);
        writeMatrix("results/matrixB_" + to_string(size) + ".txt", B);

        auto start = chrono::high_resolution_clock::now();
        auto C = multiplyMatrices(A, B);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;

        writeMatrix("results/result_" + to_string(size) + ".txt", C);

        timeResults.emplace_back(to_string(size) + "x" + to_string(size), duration.count());

        cout << "Размер: " << size << "x" << size << endl;
        cout << "  Время генерации: " << gen_duration.count() << " секунд" << endl;
        cout << "  Время умножения: " << fixed << setprecision(6) << duration.count() << " секунд" << endl;
        cout << "  Операций: " << size * size * size << " (" << 1e-9 * size * size * size << " GFlops)" << endl;
        cout << "  Производительность: " << 1e-9 * size * size * size / duration.count() << " GFlops/s" << endl;
        cout << "-----------------------------------" << endl;
    }

    writeTimeResults("results/time_results.txt", timeResults);

    cout << "Все результаты сохранены в папку 'results'" << endl;
    return 0;
}