import numpy as np
import time
import os


def verify_result(result_file, C):
    # Проверка существования файла результата
    if not os.path.isfile(result_file):
        print(f"Ошибка: файл результата не найден - {result_file}")
        return False, None  # Возвращаем False и None для обработки позже

    # Чтение результатов из файла
    with open(result_file, 'r') as file:
        lines = file.readlines()

    # Извлечение матрицы из файла
    matrix_lines = lines[1:-2]  # Предполагаем, что последние две строки не содержат матрицы
    C_cpp = np.array([list(map(int, line.split())) for line in matrix_lines])

    # Сравнение результатов
    if np.allclose(C_cpp, C):
        return True, C_cpp
    else:
        return False, C_cpp


def read_cpp_execution_time(result_file):
    # Чтение времени выполнения из файла
    if not os.path.isfile(result_file):
        print(f"Ошибка: файл результата не найден - {result_file}")
        return None

    with open(result_file, 'r') as file:
        lines = file.readlines()
        # Предполагаем, что время выполнения находится в предпоследней строке
        execution_time_line = lines[-2]  # Время выполнения перед последней строкой
        execution_time = float(execution_time_line.split(': ')[1].split()[0])
        return execution_time


# Чтение матриц из файлов
A = np.loadtxt('matrixA.txt', skiprows=1)
B = np.loadtxt('matrixB.txt', skiprows=1)

# Измерение времени выполнения
start_time = time.time()
C = np.dot(A, B)
end_time = time.time()

# Запись результата
np.savetxt('result_python.txt', C, fmt='%d')

# Вычисление объема задачи
rowsA, colsA = A.shape
colsB = B.shape[1]
volume = rowsA * colsA * colsB

# Вывод результатов
python_execution_time = end_time - start_time
results_to_save = [
    f"Время выполнения Python: {python_execution_time:.6f} секунд",
    f"Объем задачи: {volume} операций\n"
]

# Верификация результата с файлом C++
is_verified, C_cpp = verify_result('result.txt', C)

if is_verified:
    results_to_save.append("Верификация успешна: результаты совпадают!\n")
else:
    results_to_save.append("Ошибка верификации: результаты не совпадают!\n")
    results_to_save.append("Результат C++:\n")
    results_to_save.append(str(C_cpp) + "\n")
    results_to_save.append("Результат Python:\n")
    results_to_save.append(str(C) + "\n")

# Чтение времени выполнения C++
cpp_execution_time = read_cpp_execution_time('result.txt')
if cpp_execution_time is not None:
    results_to_save.append(f"Время выполнения C++: {cpp_execution_time:.6f} секунд\n")
    comparison_result = "Python " + ('медленнее' if python_execution_time > cpp_execution_time else
                                     'быстрее' if python_execution_time < cpp_execution_time else
                                     'равен') + " C++\n"
    results_to_save.append(comparison_result)

# Сохранение результатов в файл
with open('result_python.txt', 'a') as result_file:
    result_file.writelines(results_to_save)

# Печать результатов на экран
for line in results_to_save:
    print(line, end='')
