import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import minkowski

#A1
def label_encoding(df, column_name):
    unique_values = set(df[column_name])
    encoding_map = {value: idx for idx, value in enumerate(unique_values)}
    df[column_name] = df[column_name].map(encoding_map)
    return df, encoding_map

def one_hot_encoding(df, column_name):
    unique_values = df[column_name].unique()
    for value in unique_values:
        df[column_name + '_' + str(value)] = (df[column_name] == value).astype(int)
    df = df.drop(column_name, axis=1)
    return df

#A2
df = pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df, _ = label_encoding(df, col)

feature_dimensionality = df.shape[1]
print("A2 - Feature Dimensionality after encoding:", feature_dimensionality)

#A3
df_onehot = pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')
for col in categorical_cols:
    df_onehot = one_hot_encoding(df_onehot, col)
print("A3 - Feature Dimensionality after one-hot encoding:", df_onehot.shape[1])

#A4
def minkowski_distance(u, v, p):
    sum_val = 0
    for i in range(len(u)):
        sum_val += abs(u[i] - v[i]) ** p
    return sum_val ** (1/p)

#A5
u = df.iloc[0][['MntSweetProducts', 'MntGoldProds']].values
v = df.iloc[1][['MntSweetProducts', 'MntGoldProds']].values

p_values = list(range(1, 11))
distances = [minkowski_distance(u, v, p) for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, distances, marker='o', color='blue', linewidth=2, markersize=8)
plt.title("Minkowski Distance vs p", fontsize=14, fontweight='bold')
plt.xlabel("p", fontsize=12)
plt.ylabel("Minkowski Distance", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(p_values)
plt.show()

#A6
distances_custom = [minkowski_distance(u, v, p) for p in p_values]
distances_scipy = [minkowski(u, v, p) for p in p_values]

print("\nA6 - Distance Comparison:")
print("p\tCustom\t\tScipy\t\tDifference")
for p, d_custom, d_scipy in zip(p_values, distances_custom, distances_scipy):
    print(f"{p}\t{d_custom:.6f}\t{d_scipy:.6f}\t{abs(d_custom - d_scipy):.10f}")

#A7
def dot_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def euclidean_norm(vector):
    sum_sq = 0
    for val in vector:
        sum_sq += val ** 2
    return sum_sq ** 0.5

vector_a = np.array([2, 4, 6, 8, 10])
vector_b = np.array([1, 3, 5, 7, 9])

dot_custom = dot_product(vector_a, vector_b)
dot_numpy = np.dot(vector_a, vector_b)
norm_custom = euclidean_norm(vector_a)
norm_numpy = np.linalg.norm(vector_a)

print("\nA7 - Dot Product Comparison:")
print(f"Custom function: {dot_custom}")
print(f"NumPy function: {dot_numpy}")
print(f"Difference: {abs(dot_custom - dot_numpy)}")

print("\nA7 - Euclidean Norm Comparison:")
print(f"Custom function: {norm_custom:.6f}")
print(f"NumPy function: {norm_numpy:.6f}")
print(f"Difference: {abs(norm_custom - norm_numpy):.10f}")

