#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>

using namespace std;

__global__ void matrixMultiplyKernel(int* A, int* B, int* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

vector<vector<int>> generate_random_matrix(int rows, int cols) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 100);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

void write_matrix_to_file(const string& filename, const vector<vector<int>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Не удалось открыть файл для записи: " << filename << endl;
        return;
    }

    int rows = matrix.size();
    int cols = matrix[0].size();

    file << rows << " " << cols << endl;
    for (const auto& row : matrix) {
        for (int val : row) {
            file << val << " ";
        }
        file << endl;
    }
}

vector<vector<int>> multiply_matrices_cuda(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int size = A.size();

    int* flatA = new int[size * size];
    int* flatB = new int[size * size];
    int* flatC = new int[size * size];

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            flatA[i * size + j] = A[i][j];
            flatB[i * size + j] = B[i][j];
        }
    }

    int* d_A, * d_B, * d_C;
    cudaError_t err;
    err = cudaMalloc(&d_A, size * size * sizeof(int));
    if (err != cudaSuccess) {
        cerr << "Ошибка выделения памяти для d_A: " << cudaGetErrorString(err) << endl;
        delete[] flatA;
        delete[] flatB;
        delete[] flatC;
        return vector<vector<int>>();
    }

    err = cudaMalloc(&d_B, size * size * sizeof(int));
    if (err != cudaSuccess) {
        cerr << "Ошибка выделения памяти для d_B: " << cudaGetErrorString(err) << endl;
        cudaFree(d_A);
        delete[] flatA;
        delete[] flatB;
        delete[] flatC;
        return vector<vector<int>>();
    }

    err = cudaMalloc(&d_C, size * size * sizeof(int));
    if (err != cudaSuccess) {
        cerr << "Ошибка выделения памяти для d_C: " << cudaGetErrorString(err) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        delete[] flatA;
        delete[] flatB;
        delete[] flatC;
        return vector<vector<int>>();
    }

    err = cudaMemcpy(d_A, flatA, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "Ошибка копирования d_A: " << cudaGetErrorString(err) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] flatA;
        delete[] flatB;
        delete[] flatC;
        return vector<vector<int>>();
    }

    err = cudaMemcpy(d_B, flatB, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "Ошибка копирования d_B: " << cudaGetErrorString(err) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] flatA;
        delete[] flatB;
        delete[] flatC;
        return vector<vector<int>>();
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplyKernel <<<blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C, size);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Ошибка выполнения ядра: " << cudaGetErrorString(err) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] flatA;
        delete[] flatB;
        delete[] flatC;
        return vector<vector<int>>();
    }

    err = cudaMemcpy(flatC, d_C, size * size * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "Ошибка копирования результата: " << cudaGetErrorString(err) << endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] flatA;
        delete[] flatB;
        delete[] flatC;
        return vector<vector<int>>();
    }

    vector<vector<int>> C(size, vector<int>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i][j] = flatC[i * size + j];
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] flatA;
    delete[] flatB;
    delete[] flatC;

    return C;
}

int main() {

    setlocale(LC_ALL, "Russian");

    vector<int> sizes = { 100, 200, 300, 400, 500, 1000, 1500, 2000 };

    system("mkdir -p matrices");
    system("mkdir -p results");

    ofstream time_output("results/time_results.txt");
    if (!time_output.is_open()) {
        cerr << "Не удалось открыть файл для записи результатов времени" << endl;
        return 1;
    }

    time_output << "Размер матриц\tВремя выполнения (секунд)\n";

    for (int size : sizes) {
        cout << "Обработка матриц размера " << size << "x" << size << "..." << endl;

        auto A = generate_random_matrix(size, size);
        auto B = generate_random_matrix(size, size);

        string matrixA_file = "matrices/matrixA_" + to_string(size) + ".txt";
        string matrixB_file = "matrices/matrixB_" + to_string(size) + ".txt";
        write_matrix_to_file(matrixA_file, A);
        write_matrix_to_file(matrixB_file, B);

        auto start = chrono::high_resolution_clock::now();
        vector<vector<int>> C = multiply_matrices_cuda(A, B);
        auto end = chrono::high_resolution_clock::now();

        if (C.empty()) {
            cerr << "Ошибка при умножении матриц размера " << size << "x" << size << endl;
            continue;
        }

        string resultFile = "results/result_" + to_string(size) + ".txt";
        write_matrix_to_file(resultFile, C);

        chrono::duration<double> duration = end - start;
        time_output << size << "x" << size << "\t" << duration.count() << endl;
        cout << "Завершено за " << duration.count() << " секунд" << endl;
    }

    time_output.close();
    cout << "Все задачи выполнены" << endl;

    return 0;
}