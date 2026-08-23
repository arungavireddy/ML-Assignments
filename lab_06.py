# Lab Session 06: kNN Implementation with AI Tools
# 23CSE301 - Machine Learning
# File: knn_with_ai_fixed.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SECTION 1: SORTING ALGORITHMS
# Generated using ChatGPT (OpenAI) - Prompt: "Write heap sort, bubble sort, and insertion sort functions in Python"
# ============================================

def heapify(arr, n, i):
    """
    Heapify function for heap sort
    Source: ChatGPT generated with modifications
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def heap_sort(arr):
    """
    Heap sort implementation
    Source: ChatGPT generated
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    
    return arr


def bubble_sort(arr):
    """
    Bubble sort implementation with optimization
    Source: ChatGPT generated
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    
    return arr


def insertion_sort(arr):
    """
    Insertion sort implementation
    Source: ChatGPT generated
    """
    arr = arr.copy()
    
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    
    return arr


def sort_data(arr, algorithm="heap"):
    """
    Wrapper function for sorting algorithms
    Source: Modified from ChatGPT generated code
    """
    if algorithm == "heap":
        return heap_sort(arr)
    elif algorithm == "bubble":
        return bubble_sort(arr)
    elif algorithm == "insertion":
        return insertion_sort(arr)
    else:
        raise ValueError("Invalid sorting algorithm")


# ============================================
# SECTION 2: DISTANCE AND ENCODING FUNCTIONS
# Generated using Claude (Anthropic) - Prompt: "Write functions for Minkowski distance and label encoding"
# ============================================

def minkowski_distance(point1, point2, p=2):
    """
    Calculate Minkowski distance between two points
    Source: Claude AI generated
    """
    return np.sum(np.abs(point1 - point2) ** p) ** (1 / p)


def label_encoding(df, column_name):
    """
    Encode categorical variables to numeric values
    Source: Claude AI generated
    """
    unique_values = set(df[column_name])
    encoding_map = {value: idx for idx, value in enumerate(unique_values)}
    df[column_name] = df[column_name].map(encoding_map)
    return df, encoding_map


# ============================================
# SECTION 3: DATA PREPROCESSING
# Generated using GitHub Copilot - Prompt: "Create data preprocessing and imputation functions"
# ============================================

def data_imputation(df, strategy="median"):
    """
    Impute missing values in dataset
    Source: GitHub Copilot generated with modifications
    """
    df = df.copy()
    
    numeric_columns = df.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            if strategy == "mean":
                value = df[column].mean()
            elif strategy == "median":
                value = df[column].median()
            elif strategy == "mode":
                value = df[column].mode()[0]
            else:
                raise ValueError("Strategy must be mean, median or mode")
            df[column] = df[column].fillna(value)
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna('MISSING')
    
    return df


def preprocess_data(df, target_column, imputation_strategy="median", is_regression=True, scale=False):
    """
    Complete preprocessing pipeline
    Source: GitHub Copilot generated with modifications
    """
    df = df.copy()
    
    # Remove completely empty columns
    df = df.dropna(axis=1, how='all')
    
    # Remove rows where target is missing
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])
    
    # Drop non-numeric columns (specific to IRCTC dataset)
    non_numeric = ['Customer', 'Date', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 
                   'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11',
                   'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15',
                   'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18']
    
    for col in non_numeric:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Impute missing values
    df = data_imputation(df, imputation_strategy)
    
    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    for column in categorical_columns:
        if column != target_column:
            df, encoding_map = label_encoding(df, column)
    
    # Encode target if categorical and not regression
    if not is_regression and target_column in df.columns:
        if df[target_column].dtype == "object" or str(df[target_column].dtype) == "category":
            df, target_encoding = label_encoding(df, target_column)
    
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # Keep only numeric columns for X
    X = X.select_dtypes(include=[np.number])
    
    # Scale features if requested
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X.to_numpy(dtype=float)
    
    return X, y.to_numpy()


# ============================================
# SECTION 4: KNN CORE FUNCTIONS
# Generated using a combination of ChatGPT and Claude
# ============================================

def identify_neighbors(X_train, y_train, test_point, k=3, p=2, sorting_algorithm="heap"):
    """
    Identify k nearest neighbors for a test point
    Source: Combination of ChatGPT and Claude outputs
    """
    distances = []
    
    for i in range(len(X_train)):
        distance = minkowski_distance(X_train[i], test_point, p)
        distances.append((distance, i, y_train[i]))
    
    sorted_distances = sort_data(distances, sorting_algorithm)
    return sorted_distances[:k]


def majority_vote(neighbors):
    """
    Perform majority voting for classification
    Source: ChatGPT generated
    """
    class_counts = {}
    
    for distance, index, label in neighbors:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    max_votes = max(class_counts.values())
    candidates = [label for label, count in class_counts.items() if count == max_votes]
    
    for distance, index, label in neighbors:
        if label in candidates:
            return label
    
    return neighbors[0][2] if neighbors else None


def weighted_vote(neighbors):
    """
    Perform weighted voting for classification
    Source: Claude generated
    """
    class_weights = {}
    
    for distance, index, label in neighbors:
        weight = float("inf") if distance == 0 else 1 / (distance + 1e-10)
        class_weights[label] = class_weights.get(label, 0) + weight
    
    max_weight = max(class_weights.values())
    candidates = [label for label, weight in class_weights.items() if weight == max_weight]
    
    for distance, index, label in neighbors:
        if label in candidates:
            return label
    
    return neighbors[0][2] if neighbors else None


# ============================================
# SECTION 5: MY KNN CLASS
# Main implementation - Human written with AI assistance
# ============================================

class MyKNN:
    """
    Custom KNN implementation
    Developed by: Student
    AI tools used: ChatGPT, Claude, GitHub Copilot for reference and debugging
    """
    
    def __init__(self, k=3, p=2, sorting_algorithm="heap", weighted=False, regression=True):
        self.k = k
        self.p = p
        self.sorting_algorithm = sorting_algorithm
        self.weighted = weighted
        self.regression = regression
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        
        if self.k > len(self.X_train):
            self.k = len(self.X_train)
        
        return self
    
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = []
        
        for test_point in X:
            neighbors = identify_neighbors(
                self.X_train, self.y_train, test_point,
                self.k, self.p, self.sorting_algorithm
            )
            
            if self.regression:
                if self.weighted:
                    total_weight = 0
                    weighted_sum = 0
                    for distance, index, label in neighbors:
                        weight = float("inf") if distance == 0 else 1 / (distance + 1e-10)
                        weighted_sum += weight * label
                        total_weight += weight
                    prediction = weighted_sum / total_weight if total_weight > 0 else np.mean([label for _, _, label in neighbors])
                else:
                    prediction = np.mean([label for _, _, label in neighbors])
            else:
                prediction = weighted_vote(neighbors) if self.weighted else majority_vote(neighbors)
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def score(self, X, y):
        predictions = self.predict(X)
        if self.regression:
            return r2_score(y, predictions)
        else:
            return accuracy_score(y, predictions)
    
    def evaluate_regression(self, X, y):
        """
        Comprehensive evaluation for regression
        """
        predictions = self.predict(X)
        return {
            'r2': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions)
        }
    
    def evaluate_classification(self, X, y):
        """
        Comprehensive evaluation for classification
        """
        predictions = self.predict(X)
        return {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y, predictions, average='weighted', zero_division=0)
        }


# ============================================
# SECTION 6: AI-GENERATED KNN
# Complete implementation generated by AI tools
# ============================================

class AIKNN:
    """
    KNN implementation generated by AI tools
    Source: ChatGPT prompt - "Create a complete KNN class from scratch"
    """
    
    def __init__(self, k=3, metric='euclidean', regression=False):
        self.k = k
        self.metric = metric
        self.regression = regression
        self.X_train = None
        self.y_train = None
    
    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Calculate distances
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._distance(x, x_train)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance
            distances.sort(key=lambda x: x[0])
            
            # Get k nearest neighbors
            k_nearest = distances[:self.k]
            
            # Get labels/values
            values = [val for _, val in k_nearest]
            
            if self.regression:
                # For regression: average of k nearest neighbors
                prediction = np.mean(values)
            else:
                # For classification: majority vote
                prediction = max(set(values), key=values.count)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def score(self, X, y):
        predictions = self.predict(X)
        if self.regression:
            return r2_score(y, predictions)
        else:
            return accuracy_score(y, predictions)
    
    def evaluate_regression(self, X, y):
        predictions = self.predict(X)
        return {
            'r2': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions)
        }
    
    def evaluate_classification(self, X, y):
        predictions = self.predict(X)
        return {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y, predictions, average='weighted', zero_division=0)
        }


# ============================================
# SECTION 7: EVALUATION AND COMPARISON FUNCTIONS
# Generated using Claude
# ============================================

def evaluate_model_regression(model, X_test, y_test, num_runs=10):
    """
    Evaluate regression model with multiple runs for time and metrics
    Source: Claude generated
    """
    times = []
    metrics_list = []
    
    for i in range(num_runs):
        start_time = time.time()
        predictions = model.predict(X_test)
        end_time = time.time()
        
        times.append(end_time - start_time)
        
        # Store metrics
        metrics = {
            'r2': r2_score(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions)
        }
        metrics_list.append(metrics)
    
    # Average metrics
    avg_metrics = {
        'r2': np.mean([m['r2'] for m in metrics_list]),
        'mse': np.mean([m['mse'] for m in metrics_list]),
        'rmse': np.mean([m['rmse'] for m in metrics_list]),
        'mae': np.mean([m['mae'] for m in metrics_list])
    }
    
    # Average time
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_metrics, avg_time, std_time


def plot_comparison_results(k_values, custom_scores, sklearn_scores, ai_scores, metric_name="R²"):
    """
    Plot comparison of different KNN implementations
    Source: Modified from ChatGPT generated code
    """
    plt.figure(figsize=(12, 7))
    
    plt.plot(k_values, custom_scores, marker='o', linewidth=2, markersize=8, 
             label='Custom kNN', color='blue')
    plt.plot(k_values, sklearn_scores, marker='s', linewidth=2, markersize=8, 
             label='SciKit-Learn kNN', color='green')
    plt.plot(k_values, ai_scores, marker='^', linewidth=2, markersize=8, 
             label='AI-Generated kNN', color='orange')
    
    plt.xlabel('Value of k', fontsize=12)
    plt.ylabel(f'{metric_name} Score', fontsize=12)
    plt.title(f'Comparison of kNN Regressors ({metric_name} Score)', fontsize=14)
    plt.xticks(k_values)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, k in enumerate(k_values):
        if i < len(custom_scores):
            plt.annotate(f'{custom_scores[i]:.3f}', 
                        (k, custom_scores[i]), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8,
                        color='blue')
        
        if i < len(sklearn_scores):
            plt.annotate(f'{sklearn_scores[i]:.3f}', 
                        (k, sklearn_scores[i]), 
                        textcoords="offset points", 
                        xytext=(0,-15), 
                        ha='center',
                        fontsize=8,
                        color='green')
        
        if i < len(ai_scores):
            plt.annotate(f'{ai_scores[i]:.3f}', 
                        (k, ai_scores[i]), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8,
                        color='orange')
    
    plt.tight_layout()
    plt.show()


def plot_regression_comparison(k_values, custom_r2, sklearn_r2, ai_r2):
    """
    Plot R² comparison for regression models
    """
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: R² Score
    plt.subplot(1, 2, 1)
    plt.plot(k_values, custom_r2, marker='o', linewidth=2, markersize=8, 
             label='Custom kNN', color='blue')
    plt.plot(k_values, sklearn_r2, marker='s', linewidth=2, markersize=8, 
             label='Sklearn kNN', color='green')
    plt.plot(k_values, ai_r2, marker='^', linewidth=2, markersize=8, 
             label='AI kNN', color='orange')
    plt.xlabel('k value')
    plt.ylabel('R² Score')
    plt.title('R² Score Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Time Comparison
    plt.subplot(1, 2, 2)
    # We'll compute times separately
    plt.xlabel('k value')
    plt.ylabel('Time (seconds)')
    plt.title('Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================
# SECTION 8: MAIN EXECUTION
# ============================================

def main():
    """
    Main function - Human written with AI assistance for structure
    """
    DATA_FILE = "Lab Session Data.xlsx"
    SHEET_NAME = "IRCTC Stock Price"
    TARGET_COLUMN = "Price"
    IMPUTATION_STRATEGY = "median"
    SORTING_ALGORITHM = "heap"
    
    print("="*60)
    print("LAB SESSION 06: kNN Implementation Comparison (Regression)")
    print("="*60)
    
    # Load data
    try:
        df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Preprocess data - Regression mode (is_regression=True)
    X, y = preprocess_data(df, TARGET_COLUMN, IMPUTATION_STRATEGY, is_regression=True)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target statistics:")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Std: {y.std():.2f}")
    print(f"  Min: {y.min():.2f}")
    print(f"  Max: {y.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Test parameters
    max_k = min(8, len(X_train))
    k_values = list(range(1, max_k + 1))
    
    # ==========================================
    # PERFORMANCE COMPARISON (Regression)
    # ==========================================
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON (10 runs averaged)")
    print("="*60)
    
    comparison_results = []
    custom_r2_scores = []
    sklearn_r2_scores = []
    ai_r2_scores = []
    custom_times = []
    sklearn_times = []
    ai_times = []
    
    for k in k_values:
        print(f"\n--- k={k} ---")
        
        # 1. Custom KNN (Regression)
        custom_model = MyKNN(k=k, p=2, sorting_algorithm=SORTING_ALGORITHM, 
                            weighted=False, regression=True)
        custom_model.fit(X_train, y_train)
        custom_metrics, custom_time, custom_std = evaluate_model_regression(
            custom_model, X_test, y_test, num_runs=10
        )
        custom_r2_scores.append(custom_metrics['r2'])
        custom_times.append(custom_time)
        
        # 2. Sklearn KNN (Regression)
        sklearn_model = KNeighborsRegressor(n_neighbors=k)
        sklearn_model.fit(X_train, y_train)
        sklearn_metrics, sklearn_time, sklearn_std = evaluate_model_regression(
            sklearn_model, X_test, y_test, num_runs=10
        )
        sklearn_r2_scores.append(sklearn_metrics['r2'])
        sklearn_times.append(sklearn_time)
        
        # 3. AI-Generated KNN (Regression)
        ai_model = AIKNN(k=k, metric='euclidean', regression=True)
        ai_model.fit(X_train, y_train)
        ai_metrics, ai_time, ai_std = evaluate_model_regression(
            ai_model, X_test, y_test, num_runs=10
        )
        ai_r2_scores.append(ai_metrics['r2'])
        ai_times.append(ai_time)
        
        # Store results
        comparison_results.append({
            'k': k,
            'custom_r2': custom_metrics['r2'],
            'custom_rmse': custom_metrics['rmse'],
            'custom_mae': custom_metrics['mae'],
            'custom_time': custom_time,
            'custom_time_std': custom_std,
            'sklearn_r2': sklearn_metrics['r2'],
            'sklearn_rmse': sklearn_metrics['rmse'],
            'sklearn_mae': sklearn_metrics['mae'],
            'sklearn_time': sklearn_time,
            'sklearn_time_std': sklearn_std,
            'ai_r2': ai_metrics['r2'],
            'ai_rmse': ai_metrics['rmse'],
            'ai_mae': ai_metrics['mae'],
            'ai_time': ai_time,
            'ai_time_std': ai_std
        })
        
        # Print summary
        print(f"Custom KNN    - R²: {custom_metrics['r2']:.4f}, RMSE: {custom_metrics['rmse']:.2f}, Time: {custom_time:.4f}s")
        print(f"Sklearn KNN   - R²: {sklearn_metrics['r2']:.4f}, RMSE: {sklearn_metrics['rmse']:.2f}, Time: {sklearn_time:.4f}s")
        print(f"AI KNN        - R²: {ai_metrics['r2']:.4f}, RMSE: {ai_metrics['rmse']:.2f}, Time: {ai_time:.4f}s")
    
    # Create comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE (Regression Metrics)")
    print("="*60)
    
    results_df = pd.DataFrame(comparison_results)
    print("\nR² Scores:")
    print(results_df[['k', 'custom_r2', 'sklearn_r2', 'ai_r2']].to_string())
    print("\nRMSE Scores:")
    print(results_df[['k', 'custom_rmse', 'sklearn_rmse', 'ai_rmse']].to_string())
    print("\nMAE Scores:")
    print(results_df[['k', 'custom_mae', 'sklearn_mae', 'ai_mae']].to_string())
    print("\nTime (seconds):")
    print(results_df[['k', 'custom_time', 'sklearn_time', 'ai_time']].to_string())
    
    # Plot results
    plot_comparison_results(
        k_values, 
        results_df['custom_r2'].values,
        results_df['sklearn_r2'].values,
        results_df['ai_r2'].values,
        "R² Score"
    )
    
    # Best models
    print("\n" + "="*60)
    print("BEST PERFORMING MODELS")
    print("="*60)
    
    best_custom_idx = results_df['custom_r2'].idxmax()
    best_sklearn_idx = results_df['sklearn_r2'].idxmax()
    best_ai_idx = results_df['ai_r2'].idxmax()
    
    print(f"Best Custom KNN:    k={results_df.loc[best_custom_idx, 'k']}, "
          f"R²={results_df.loc[best_custom_idx, 'custom_r2']:.4f}, "
          f"RMSE={results_df.loc[best_custom_idx, 'custom_rmse']:.2f}")
    print(f"Best Sklearn KNN:   k={results_df.loc[best_sklearn_idx, 'k']}, "
          f"R²={results_df.loc[best_sklearn_idx, 'sklearn_r2']:.4f}, "
          f"RMSE={results_df.loc[best_sklearn_idx, 'sklearn_rmse']:.2f}")
    print(f"Best AI KNN:        k={results_df.loc[best_ai_idx, 'k']}, "
          f"R²={results_df.loc[best_ai_idx, 'ai_r2']:.4f}, "
          f"RMSE={results_df.loc[best_ai_idx, 'ai_rmse']:.2f}")
    
    # Save results to CSV
    results_df.to_csv("knn_comparison_results.csv", index=False)
    print("\nResults saved to 'knn_comparison_results.csv'")


if __name__ == "__main__":
    main()