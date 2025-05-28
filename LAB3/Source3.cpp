#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <mpi.h>

using namespace std;

vector<vector<int>> generateRandomMatrix(int rows, int cols, int seed = 0) {
    vector<vector<int>> matrix(rows, vector<int>(cols));
    mt19937 gen(seed == 0 ? random_device{}() : seed);
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

void parallelMatrixMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B,
    vector<vector<int>>& C, int rank, int size) {
    int rowsA = A.size();
    int colsB = B[0].size();
    int colsA = A[0].size();

    int rows_per_process = rowsA / size;
    int extra_rows = rowsA % size;

    int start_row = rank * rows_per_process + min(rank, extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < colsB; ++j) {
            int sum = 0;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    if (rank != 0) {

        for (int i = start_row; i < end_row; ++i) {
            MPI_Send(C[i].data(), colsB, MPI_INT, 0, i, MPI_COMM_WORLD);
        }
    }
    else {

        for (int p = 1; p < size; ++p) {
            int p_start = p * rows_per_process + min(p, extra_rows);
            int p_end = p_start + rows_per_process + (p < extra_rows ? 1 : 0);

            for (int i = p_start; i < p_end; ++i) {
                MPI_Recv(C[i].data(), colsB, MPI_INT, p, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> sizes = { 100, 200, 300, 400, 500, 1000, 1500, 3000 };
    vector<pair<string, double>> timeResults;

    if (rank == 0) {
        system("if not exist results mkdir results");

    }

    for (int matrix_size : sizes) {
        vector<vector<int>> A, B, C(matrix_size, vector<int>(matrix_size));

        if (rank == 0) {
            A = generateRandomMatrix(matrix_size, matrix_size);
            B = generateRandomMatrix(matrix_size, matrix_size, 42);

            writeMatrix("results/matrixA_" + to_string(matrix_size) + ".txt", A);
            writeMatrix("results/matrixB_" + to_string(matrix_size) + ".txt", B);
        }

        if (rank == 0) {
            for (int p = 1; p < size; ++p) {
                for (const auto& row : B) {
                    MPI_Send(row.data(), matrix_size, MPI_INT, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            B.resize(matrix_size, vector<int>(matrix_size));
            for (auto& row : B) {
                MPI_Recv(row.data(), matrix_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        if (rank == 0) {
            int rows_per_process = matrix_size / size;
            int extra_rows = matrix_size % size;

            for (int p = 1; p < size; ++p) {
                int start_row = p * rows_per_process + min(p, extra_rows);
                int end_row = start_row + rows_per_process + (p < extra_rows ? 1 : 0);

                for (int i = start_row; i < end_row; ++i) {
                    MPI_Send(A[i].data(), matrix_size, MPI_INT, p, 1, MPI_COMM_WORLD);
                }
            }
        }
        else {
            int rows_per_process = matrix_size / size;
            int extra_rows = matrix_size % size;
            int my_rows = rows_per_process + (rank < extra_rows ? 1 : 0);

            A.resize(my_rows, vector<int>(matrix_size));
            for (auto& row : A) {
                MPI_Recv(row.data(), matrix_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        if (rank == 0) {
            parallelMatrixMultiply(A, B, C, rank, size);
        }
        else {
            vector<vector<int>> local_C(A.size(), vector<int>(matrix_size));
            parallelMatrixMultiply(A, B, local_C, rank, size);
        }

        double end_time = MPI_Wtime();
        double duration = end_time - start_time;

        double max_duration;
        MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            timeResults.emplace_back(to_string(matrix_size) + "x" + to_string(matrix_size), max_duration);
            writeMatrix("results/result_" + to_string(matrix_size) + ".txt", C);

            cout << "Size: " << matrix_size << "x" << matrix_size << endl;
            cout << "Time: " << fixed << setprecision(6) << max_duration << " seconds" << endl;
            cout << "Operations: " << matrix_size * matrix_size * matrix_size << endl;
            cout << "Performance: " << 1e-9 * matrix_size * matrix_size * matrix_size / max_duration << " GFlops/s" << endl;
            cout << "-----------------------------------" << endl;
        }
    }

    if (rank == 0) {
        writeTimeResults("results/time_results.txt", timeResults);
        cout << "All results saved to 'results' directory" << endl;
    }

    MPI_Finalize();
    return 0;
}