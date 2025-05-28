import os
import numpy as np

matrix_dir = "../LAB3/results"
result_dir = "../LAB3/results"
verification_file = "../LAB3/results/verification_results.txt"


def read_matrix_from_file(filename):
    with open(filename, "r") as file:
        rows, cols = map(int, file.readline().split())
        matrix = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            matrix[i] = [int(x) for x in file.readline().split()]
    return matrix


def check_results():
    with open(verification_file, "w", encoding="utf-8") as f:
        f.write("Размер матриц\tРезультат проверки\n")
        for size in [100, 200, 300, 400, 500, 1000, 1500, 3000]:
            matrix_a_file = os.path.join(matrix_dir, f"matrixA_{size}.txt")
            matrix_b_file = os.path.join(matrix_dir, f"matrixB_{size}.txt")
            result_file = os.path.join(result_dir, f"result_{size}.txt")

            matrix_a = read_matrix_from_file(matrix_a_file)
            matrix_b = read_matrix_from_file(matrix_b_file)
            result = read_matrix_from_file(result_file)

            expected_result = np.matmul(matrix_a, matrix_b)

            if np.array_equal(result, expected_result):
                status = "Корректно"
            else:
                status = "Некорректно"
            f.write(f"{size}x{size}\t{status}\n")
            print(f"Результат для матриц размера {size}x{size} - {status}")


def read_time_results():
    matrix_sizes = []
    execution_times = []
    with open("../LAB3/results/time_results.txt", "r") as file:
        next(file)
        for line in file:
            parts = line.strip().split("\t")
            size_info = parts[0]
            time = parts[1]

            if "x" in size_info:
                rows, cols = map(int, size_info.split("x"))
                matrix_sizes.append(rows)
            else:
                matrix_sizes.append(int(size_info))
            execution_times.append(float(time))

    return matrix_sizes, execution_times


if __name__ == "__main__":
    matrix_sizes, execution_times = read_time_results()
    check_results()