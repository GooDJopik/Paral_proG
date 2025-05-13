#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>

using namespace std;

vector<vector<int>> generateRandomMatrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 10);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = dis(gen);

    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<int>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
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

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

void writeTimeResults(const string& filename, const vector<pair<string, double>>& results) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening time results file: " << filename << endl;
        return;
    }

    file << "Размер матриц\tВремя выполнения (секунд)" << endl;

    for (const auto& result : results) {
        file << result.first << "\t" << fixed << setprecision(6) << result.second << endl;
    }
}

int main() {
    vector<int> sizes = { 100, 200, 300, 400, 500, 1000 };
    vector<pair<string, double>> timeResults;

    system("mkdir -p results");

    for (int size : sizes) {
        auto A = generateRandomMatrix(size, size);
        auto B = generateRandomMatrix(size, size);

        writeMatrix("results/matrixA_" + to_string(size) + ".txt", A);
        writeMatrix("results/matrixB_" + to_string(size) + ".txt", B);

        auto start = chrono::high_resolution_clock::now();
        auto C = multiplyMatrices(A, B);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;

        writeMatrix("results/result_" + to_string(size) + ".txt", C);

        timeResults.emplace_back(to_string(size) + "x" + to_string(size), duration.count());

        cout << "Size: " << size << "x" << size << endl;
        cout << "Time: " << fixed << setprecision(6) << duration.count() << " seconds" << endl;
        cout << "Operations: " << size * size * size << endl;
        cout << "-----------------------------------" << endl;
    }

    writeTimeResults("results/time_results.txt", timeResults);

    cout << "All results saved to 'results' directory" << endl;
    return 0;
}