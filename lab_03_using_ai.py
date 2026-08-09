import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

# A1 - Function to convert categorical labels to numeric values
def label_encoding(df, column_name):
    # Get all unique values from the specified column
    unique_values = set(df[column_name])
    # Create a mapping dictionary: each unique value gets a numeric index
    encoding_map = {value: idx for idx, value in enumerate(unique_values)}
    # Replace original values with their numeric equivalents
    df[column_name] = df[column_name].map(encoding_map)
    # Return updated dataframe and the mapping for reference
    return df, encoding_map

# A1 - Function to create dummy/one-hot encoded variables
def one_hot_encoding(df, column_name):
    # Create dummy variables using pandas get_dummies function
    dummy_df = pd.get_dummies(df[column_name], prefix=column_name)
    # Remove the original column and concatenate the dummy columns
    df = pd.concat([df.drop(column_name, axis=1), dummy_df], axis=1)
    return df

# A2 - Load the Excel file and apply label encoding
df = pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')
# Identify all categorical columns (object type)
categorical_cols = df.select_dtypes(include=['object']).columns

# Apply label encoding to each categorical column
for col in categorical_cols:
    df, _ = label_encoding(df, col)

# A2 - Print the new feature dimensionality after encoding
feature_dimensionality = df.shape[1]
print("A2 - Feature Dimensionality after encoding:", feature_dimensionality)

# A3 - Apply one-hot encoding to the same dataset
df_onehot = pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')
for col in categorical_cols:
    df_onehot = one_hot_encoding(df_onehot, col)
# Print the feature dimensionality after one-hot encoding
print("A3 - Feature Dimensionality after one-hot encoding:", df_onehot.shape[1])

# A4 - Function to compute Minkowski distance between two vectors
def minkowski_distance(u, v, p):
    # Initialize sum to zero
    sum_val = 0
    # Iterate through each dimension of the vectors
    for i in range(len(u)):
        # Add absolute difference raised to power p
        sum_val += abs(u[i] - v[i]) ** p
    # Return the p-th root of the sum
    return sum_val ** (1/p)

# A5 - Extract two data points for Minkowski distance calculation
# Select first row's values for 'MntSweetProducts' and 'MntGoldProds'
u = df.iloc[0][['MntSweetProducts', 'MntGoldProds']].values
# Select second row's values for 'MntSweetProducts' and 'MntGoldProds'
v = df.iloc[1][['MntSweetProducts', 'MntGoldProds']].values

# Create list of p values from 1 to 10
p_values = list(range(1, 11))
# Calculate Minkowski distance for each p value
distances = [minkowski_distance(u, v, p) for p in p_values]

# A5 - Plot the Minkowski distance vs p values
plt.figure(figsize=(10, 6))
plt.plot(p_values, distances, marker='o', color='blue', linewidth=2, markersize=8)
plt.title("Minkowski Distance vs p", fontsize=14, fontweight='bold')
plt.xlabel("p", fontsize=12)
plt.ylabel("Minkowski Distance", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(p_values)
plt.show()

# A6 - Compare custom Minkowski function with scipy's implementation
distances_custom = [minkowski_distance(u, v, p) for p in p_values]
distances_scipy = [minkowski(u, v, p) for p in p_values]

print("\nA6 - Distance Comparison:")
print("p\tCustom\t\tScipy\t\tDifference")
for p, d_custom, d_scipy in zip(p_values, distances_custom, distances_scipy):
    print(f"{p}\t{d_custom:.6f}\t{d_scipy:.6f}\t{abs(d_custom - d_scipy):.10f}")

# A7 - Custom function to compute dot product
def dot_product(a, b):
    result = 0
    # Iterate through elements of both vectors
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# A7 - Custom function to compute Euclidean norm
def euclidean_norm(vector):
    sum_sq = 0
    # Sum of squares of all elements
    for val in vector:
        sum_sq += val ** 2
    # Return square root of sum of squares
    return sum_sq ** 0.5

# A7 - Test vectors for dot product and norm calculations
vector_a = np.array([2, 4, 6, 8, 10])
vector_b = np.array([1, 3, 5, 7, 9])

# Calculate dot product using custom and numpy functions
dot_custom = dot_product(vector_a, vector_b)
dot_numpy = np.dot(vector_a, vector_b)
# Calculate Euclidean norm using custom and numpy functions
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

# A8 - Function to calculate mean of data (handles DataFrame or array)
def calculate_mean(data):
    # Check if data is a DataFrame
    if isinstance(data, pd.DataFrame):
        means = []
        # Iterate through each column
        for col in data.columns:
            col_sum = 0
            count = 0
            # Sum valid (non-NaN) values and count them
            for val in data[col]:
                if pd.notna(val):
                    col_sum += val
                    count += 1
            # Calculate mean, avoid division by zero
            means.append(col_sum / count if count > 0 else 0)
        return np.array(means)
    else:
        # Handle array/list input
        total = 0
        count = 0
        for val in data:
            if pd.notna(val):
                total += val
                count += 1
        return total / count if count > 0 else 0

# A8 - Function to calculate variance
def calculate_variance(data, mean_val=None):
    # Calculate mean if not provided
    if mean_val is None:
        mean_val = calculate_mean(data)
    
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        variances = []
        # Iterate through columns
        for idx, col in enumerate(data.columns):
            sum_sq_diff = 0
            count = 0
            # Sum of squared differences from mean
            for val in data[col]:
                if pd.notna(val):
                    sum_sq_diff += (val - mean_val[idx]) ** 2
                    count += 1
            # Calculate variance (population variance)
            variances.append(sum_sq_diff / count if count > 0 else 0)
        return np.array(variances)
    else:
        # Handle array/list input
        sum_sq_diff = 0
        count = 0
        for val in data:
            if pd.notna(val):
                sum_sq_diff += (val - mean_val) ** 2
                count += 1
        return sum_sq_diff / count if count > 0 else 0

# A8 - Function to calculate standard deviation
def calculate_std_dev(data, variance_val=None):
    # Calculate variance if not provided
    if variance_val is None:
        variance_val = calculate_variance(data)
    
    # Return square root of variance
    if isinstance(data, pd.DataFrame):
        return np.sqrt(variance_val)
    else:
        return variance_val ** 0.5

# A8 - Function to calculate all statistics (mean, variance, std)
def calculate_stats(data):
    mean_val = calculate_mean(data)
    variance_val = calculate_variance(data, mean_val)
    std_val = calculate_std_dev(data, variance_val)
    return mean_val, variance_val, std_val

# A8 - Select numeric data and handle missing values
numeric_data = df.select_dtypes(include=[np.number])
numeric_data_clean = numeric_data.dropna()
# Calculate statistics using custom functions
mean_custom, var_custom, std_custom = calculate_stats(numeric_data_clean)

print("\nA8 - Custom Statistics for first 5 features:")
for i, col in enumerate(numeric_data_clean.columns[:5]):
    print(f"{col}: Mean={mean_custom[i]:.4f}, Var={var_custom[i]:.4f}, Std={std_custom[i]:.4f}")

# A9 - Calculate statistics using NumPy for comparison
mean_numpy = np.mean(numeric_data_clean, axis=0)
std_numpy = np.std(numeric_data_clean, axis=0)

print("\nA9 - Comparison for first 5 features:")
print("\nMean Comparison:")
for i, col in enumerate(numeric_data_clean.columns[:5]):
    print(f"{col}: Custom={mean_custom[i]:.6f}, NumPy={mean_numpy[i]:.6f}, Diff={abs(mean_custom[i] - mean_numpy[i]):.10f}")

print("\nStandard Deviation Comparison:")
for i, col in enumerate(numeric_data_clean.columns[:5]):
    print(f"{col}: Custom={std_custom[i]:.6f}, NumPy={std_numpy[i]:.6f}, Diff={abs(std_custom[i] - std_numpy[i]):.10f}")

# A10 - Select the first feature for histogram visualization
selected_feature = numeric_data_clean.columns[0]
feature_values = numeric_data_clean[selected_feature].values

# A10 - Create histogram with 15 bins
plt.figure(figsize=(10, 6))
hist_data, bins, patches = plt.hist(feature_values, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
plt.title(f"Histogram of {selected_feature}", fontsize=14, fontweight='bold')
plt.xlabel(selected_feature, fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
# Add vertical line for mean value
plt.axvline(calculate_mean(feature_values), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {calculate_mean(feature_values):.2f}')
plt.legend()
plt.show()

# A10 - Calculate and display statistics for the selected feature
feature_mean = calculate_mean(feature_values) 
feature_var = calculate_variance(feature_values, feature_mean)

print(f"\nA10 - Statistics for {selected_feature}:")
print(f"Mean: {feature_mean:.4f}")
print(f"Variance: {feature_var:.4f}")
print(f"Standard Deviation: {feature_var ** 0.5:.4f}")
print(f"Number of data points: {len(feature_values)}")
print(f"Histogram bins: {len(bins)-1}")
print(f"Data range: {feature_values.min():.2f} to {feature_values.max():.2f}")

# A11 - Custom K-means clustering implementation
def kmeans(X, k, max_iters=100, tol=1e-4):
    # Set seed for reproducibility
    np.random.seed(42)
    n_samples, n_features = X.shape
    
    # Initialize centroids by randomly selecting k data points
    initial_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[initial_indices].copy()
    
    # Iterate until convergence or max iterations
    for iteration in range(max_iters):
        # Calculate distance from each point to each centroid
        distances = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                distances[i, j] = minkowski_distance(X[i], centroids[j], 2)
        
        # Assign each point to nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Update centroids as mean of points in each cluster
        new_centroids = np.zeros((k, n_features))
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                # Keep old centroid if cluster is empty
                new_centroids[j] = centroids[j]
        
        # Calculate how much centroids have shifted
        centroid_shift = 0
        for j in range(k):
            centroid_shift += minkowski_distance(centroids[j], new_centroids[j], 2)
        
        # Update centroids
        centroids = new_centroids
        
        # Check convergence
        if centroid_shift < tol:
            break
    
    return centroids, labels

# A11 - Prepare data and run K-means
X_sample = numeric_data_clean.values[:200]
k = 3
centroids, labels = kmeans(X_sample, k)

print("\nA11 - K-means Clustering Results:")
print(f"Number of clusters: {k}")
print(f"Number of samples: {len(X_sample)}")
print(f"Number of features: {X_sample.shape[1]}")
print(f"Iterations completed: {len(np.unique(labels))}")
print("\nCluster centroids (first 5 features):")
for i in range(k):
    print(f"Cluster {i+1}: {centroids[i][:5]}")

print(f"\nCluster distribution:")
for i in range(k):
    print(f"Cluster {i+1}: {np.sum(labels == i)} samples")

print("\nAll tasks completed successfully!")