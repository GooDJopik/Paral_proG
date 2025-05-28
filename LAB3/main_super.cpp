#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

vector<vector<int> > generate_random_matrix(int rows, int cols, int min = 0, int max = 99) {
    vector<vector<int> > matrix(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % (max - min + 1) + min;
        }
    }
    return matrix;
}

double parallel_matrix_multiply(
    const vector<vector<int> >& matrix_1,
    const vector<vector<int> >& matrix_2,
    vector<vector<int> >& matrix_result,
    int rank, int size) {

    int n = matrix_1.size();
    matrix_result.resize(n, vector<int>(n, 0));

    double start_time = MPI_Wtime();

    int rows_per_process = n / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? n : start_row + rows_per_process;

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                matrix_result[i][j] += matrix_1[i][k] * matrix_2[k][j];
            }
        }
    }

    if (rank != 0) {
        for (int i = start_row; i < end_row; ++i) {
            MPI_Send(&matrix_result[i][0], n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    } else {
        for (int proc = 1; proc < size; ++proc) {
            int proc_start = proc * rows_per_process;
            int proc_end = (proc == size - 1) ? n : proc_start + rows_per_process;

            for (int i = proc_start; i < proc_end; ++i) {
                MPI_Recv(&matrix_result[i][0], n, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    return MPI_Wtime() - start_time;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        srand(time(NULL));
    }

    int matrix_sizes[] = {100, 200, 300, 400, 500, 1000, 1500, 2000};
    int num_sizes = sizeof(matrix_sizes) / sizeof(matrix_sizes[0]);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int s = 0; s < num_sizes; ++s) {
        int matrix_size = matrix_sizes[s];
        vector<vector<int> > matrix_1, matrix_2, result;

        if (rank == 0) {
            matrix_1 = generate_random_matrix(matrix_size, matrix_size);
            matrix_2 = generate_random_matrix(matrix_size, matrix_size);
        }

        if (rank == 0) {
            for (int proc = 1; proc < size; ++proc) {
                for (int i = 0; i < matrix_size; ++i) {
                    MPI_Send(&matrix_1[i][0], matrix_size, MPI_INT, proc, 0, MPI_COMM_WORLD);
                }
                for (int i = 0; i < matrix_size; ++i) {
                    MPI_Send(&matrix_2[i][0], matrix_size, MPI_INT, proc, 1, MPI_COMM_WORLD);
                }
            }
        } else {
            matrix_1.resize(matrix_size, vector<int>(matrix_size));
            matrix_2.resize(matrix_size, vector<int>(matrix_size));

            for (int i = 0; i < matrix_size; ++i) {
                MPI_Recv(&matrix_1[i][0], matrix_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&matrix_2[i][0], matrix_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double duration = parallel_matrix_multiply(matrix_1, matrix_2, result, rank, size);

        if (rank == 0) {
            cout << "SIZE: " << matrix_size << "x" << matrix_size << ", ";
            cout << "Proceses: " << size << ", ";
            cout << "Time: " << duration << " seconds" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}