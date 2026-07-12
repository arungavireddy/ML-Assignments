#1

def find_pairs(lst):
    result = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] + lst[j] == 10:
                result.append((lst[i], lst[j]))
    return result

l1 = [2, 7, 4, 1, 3, 6]

result = find_pairs(l1)
print(result)

# 2

def find_range(lst):
    min_val = lst[0]
    max_val = lst[0]

    for num in lst:
        if num < min_val:
            min_val = num
        if num > max_val:
            max_val = num

    return min_val, max_val

l2 = [5, 3, 8, 1, 0, 4]

minimum, maximum = find_range(l2)
print("Range of the given list is:", minimum, "-", maximum)

# 3

import copy

def multiply_matrices(A, B, n):
    result = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]

    return result


def matrix_power(matrix, n, m):
    if m == 0:
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    result = copy.deepcopy(matrix)

    for _ in range(1, m):
        result = multiply_matrices(result, matrix, n)

    return result


n = int(input("Enter the size of the matrix: "))

print("Enter the elements of square matrix:")
matrix = []

for i in range(n):
    row = []
    for j in range(n):
        element = int(input(f"Enter element [{i}][{j}]: "))
        row.append(element)
    matrix.append(row)

m = int(input("Enter a number (power m): "))

result = matrix_power(matrix, n, m)

print(f"\nResult of matrix A to the power {m} is:")
for row in result:
    print(row)
    
# 4

def highest_occurring_char(word):
    charac_count = {}

    for letter in word:
        charac_count[letter] = charac_count.get(letter, 0) + 1

    max_char = max(charac_count, key=charac_count.get)
    max_count = charac_count[max_char]

    return max_char, max_count


word = input("Enter a string: ")

character, count = highest_occurring_char(word)

print("The highest occurring letter is:", character)
print("Its count is:", count)

#5

import random

def calculate_statistics():
    random_numbers = [random.randint(1, 10) for _ in range(25)]

    total_sum = sum(random_numbers)
    total_count = len(random_numbers)

    calculated_mean = total_sum / total_count

    sorted_numbers = sorted(random_numbers)

    middle_index = total_count // 2

    if total_count % 2 != 0:
        calculated_median = sorted_numbers[middle_index]
    else:
        calculated_median = (sorted_numbers[middle_index - 1] + sorted_numbers[middle_index]) / 2

    frequency_map = {}

    for num in random_numbers:
        frequency_map[num] = frequency_map.get(num, 0) + 1

    max_frequency = max(frequency_map.values())

    calculated_mode = []

    for num, count in frequency_map.items():
        if count == max_frequency:
            calculated_mode.append(num)

    return random_numbers, sorted_numbers, calculated_mean, calculated_median, calculated_mode

numbers, sorted_numbers, mean, median, mode = calculate_statistics()

print(f"Mean:   {mean:.2f}")
print(f"Median: {median}")
print(f"Mode:   {mode}")