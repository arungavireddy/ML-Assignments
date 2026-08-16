import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#A1

def heapify(arr, n, i):
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
    arr = arr.copy()
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

    return arr


def bubble_sort(arr):
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
    if algorithm == "heap":
        return heap_sort(arr)
    elif algorithm == "bubble":
        return bubble_sort(arr)
    elif algorithm == "insertion":
        return insertion_sort(arr)
    else:
        raise ValueError("Invalid sorting algorithm")


def minkowski_distance(point1, point2, p=2):
    return np.sum(np.abs(point1 - point2) ** p) ** (1 / p)


def label_encoding(df, column_name):
    unique_values = set(df[column_name])
    encoding_map = {value: idx for idx, value in enumerate(unique_values)}
    df[column_name] = df[column_name].map(encoding_map)
    return df, encoding_map


def data_imputation(df, strategy="median"):
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


def identify_neighbors(X_train, y_train, test_point, k=3, p=2, sorting_algorithm="heap"):
    distances = []

    for i in range(len(X_train)):
        distance = minkowski_distance(X_train[i], test_point, p)
        distances.append((distance, i, y_train[i]))

    sorted_distances = sort_data(distances, sorting_algorithm)
    return sorted_distances[:k]


def majority_vote(neighbors):
    class_counts = {}

    for distance, index, label in neighbors:
        class_counts[label] = class_counts.get(label, 0) + 1

    max_votes = max(class_counts.values())
    candidates = [label for label, count in class_counts.items() if count == max_votes]

    for distance, index, label in neighbors:
        if label in candidates:
            return label

    return neighbors[0][2] if neighbors else None


#A2

def weighted_vote(neighbors):
    class_weights = {}

    for distance, index, label in neighbors:
        weight = float("inf") if distance == 0 else 1 / distance
        class_weights[label] = class_weights.get(label, 0) + weight

    max_weight = max(class_weights.values())
    candidates = [label for label, weight in class_weights.items() if weight == max_weight]

    for distance, index, label in neighbors:
        if label in candidates:
            return label

    return neighbors[0][2] if neighbors else None


#A7

class MyKNN:
    def __init__(self, k=3, p=2, sorting_algorithm="heap", weighted=False, regression=False):
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
            return np.sum(predictions == np.asarray(y)) / len(y)

#A3

def preprocess_data(df, target_column, imputation_strategy="median", is_regression=True):
    df = df.copy()

    df = df.dropna(axis=1, how='all')

    if target_column in df.columns:
        df = df.dropna(subset=[target_column])

    non_numeric = ['Customer', 'Date', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
                   'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11',
                   'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15',
                   'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18']

    for col in non_numeric:
        if col in df.columns:
            df = df.drop(col, axis=1)

    df = data_imputation(df, imputation_strategy)

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    for column in categorical_columns:
        if column != target_column:
            df, encoding_map = label_encoding(df, column)

    if not is_regression and target_column in df.columns:
        if df[target_column].dtype == "object" or str(df[target_column].dtype) == "category":
            df, target_encoding = label_encoding(df, target_column)

    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    X = X.select_dtypes(include=[np.number])

    return X.to_numpy(dtype=float), y.to_numpy()

#A8
def plot_results(k_values, custom_scores, sklearn_scores, weighted_scores, metric_name="R²"):
    plt.figure(figsize=(12, 7))

    plt.plot(k_values, custom_scores, marker='o', linewidth=2, markersize=8,
             label='Custom kNN', color='blue')
    plt.plot(k_values, sklearn_scores, marker='s', linewidth=2, markersize=8,
             label='SciKit-Learn kNN', color='green')
    plt.plot(k_values, weighted_scores, marker='^', linewidth=2, markersize=8,
             label='Weighted kNN', color='red')

    plt.xlabel('Value of k', fontsize=12)
    plt.ylabel(f'{metric_name} Score', fontsize=12)
    plt.title(f'Comparison of kNN Regressors ({metric_name} Score)', fontsize=14)
    plt.xticks(k_values)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

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

        if i < len(weighted_scores):
            plt.annotate(f'{weighted_scores[i]:.3f}',
                        (k, weighted_scores[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8,
                        color='red')

    plt.tight_layout()
    plt.show()


def main():
    DATA_FILE = "Lab Session Data.xlsx"
    SHEET_NAME = "IRCTC Stock Price"
    TARGET_COLUMN = "Price"
    IMPUTATION_STRATEGY = "median"
    SORTING_ALGORITHM = "heap"

    try:
        df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")
    print("\nFirst 5 rows:")
    print(df.head())

    df.columns = df.columns.str.strip()

    X, y = preprocess_data(df, TARGET_COLUMN, IMPUTATION_STRATEGY, is_regression=True)

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTarget statistics:")
    print(f"Mean: {y.mean():.2f}")
    print(f"Std: {y.std():.2f}")
    print(f"Min: {y.min():.2f}")
    print(f"Max: {y.max():.2f}")

    test_size = 0.2 if len(X) > 10 else 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    k_test = min(2, len(X_train))

    #A4
    sklearn_model = KNeighborsRegressor(n_neighbors=k_test)
    sklearn_model.fit(X_train, y_train)
    #A5
    sklearn_r2 = sklearn_model.score(X_test, y_test)
    #A6
    sklearn_pred = sklearn_model.predict(X_test)

    print(f"\n{'='*50}")
    print(f"SciKit-Learn kNN (k={k_test})")
    print(f"{'='*50}")
    print(f"R² Score: {sklearn_r2:.4f}")
    print(f"Predictions: {sklearn_pred}")

    custom_model = MyKNN(k=k_test, p=2, sorting_algorithm=SORTING_ALGORITHM,
                         weighted=False, regression=True)
    custom_model.fit(X_train, y_train)
    custom_r2 = custom_model.score(X_test, y_test)
    custom_pred = custom_model.predict(X_test)

    print(f"\n{'='*50}")
    print(f"Custom kNN (k={k_test}, {SORTING_ALGORITHM} sort)")
    print(f"{'='*50}")
    print(f"R² Score: {custom_r2:.4f}")
    print(f"Predictions: {custom_pred}")

    weighted_model = MyKNN(k=k_test, p=2, sorting_algorithm=SORTING_ALGORITHM,
                          weighted=True, regression=True)
    weighted_model.fit(X_train, y_train)
    weighted_r2 = weighted_model.score(X_test, y_test)
    weighted_pred = weighted_model.predict(X_test)

    print(f"\n{'='*50}")
    print(f"Weighted kNN (k={k_test}, {SORTING_ALGORITHM} sort)")
    print(f"{'='*50}")
    print(f"R² Score: {weighted_r2:.4f}")
    print(f"Predictions: {weighted_pred}")

    max_k = min(8, len(X_train))
    k_values = list(range(1, max_k + 1))

    custom_scores = []
    sklearn_scores = []
    weighted_scores = []

    print(f"\n{'='*50}")
    print(f"Comparing different k values (1 to {max_k})")
    print(f"{'='*50}")

    for k in k_values:
        model = MyKNN(k=k, p=2, sorting_algorithm=SORTING_ALGORITHM,
                     weighted=False, regression=True)
        model.fit(X_train, y_train)
        custom_score = model.score(X_test, y_test)
        custom_scores.append(custom_score)

        sk_model = KNeighborsRegressor(n_neighbors=k)
        sk_model.fit(X_train, y_train)
        sk_score = sk_model.score(X_test, y_test)
        sklearn_scores.append(sk_score)

        w_model = MyKNN(k=k, p=2, sorting_algorithm=SORTING_ALGORITHM,
                       weighted=True, regression=True)
        w_model.fit(X_train, y_train)
        weighted_score = w_model.score(X_test, y_test)
        weighted_scores.append(weighted_score)

        print(f"k={k}: Custom={custom_score:.4f}, Sklearn={sk_score:.4f}, Weighted={weighted_score:.4f}")

    results = pd.DataFrame({
        "k": k_values,
        "Custom kNN": custom_scores,
        "SciKit-Learn kNN": sklearn_scores,
        "Weighted kNN": weighted_scores
    })

    print(f"\n{'='*50}")
    print("Summary Results")
    print(f"{'='*50}")
    print(results)

    best_custom_idx = np.argmax(custom_scores)
    best_sklearn_idx = np.argmax(sklearn_scores)
    best_weighted_idx = np.argmax(weighted_scores)

    print(f"\n{'='*50}")
    print("Best Models")
    print(f"{'='*50}")
    print(f"Best Custom kNN: k={k_values[best_custom_idx]}, R²={custom_scores[best_custom_idx]:.4f}")
    print(f"Best Sklearn kNN: k={k_values[best_sklearn_idx]}, R²={sklearn_scores[best_sklearn_idx]:.4f}")
    print(f"Best Weighted kNN: k={k_values[best_weighted_idx]}, R²={weighted_scores[best_weighted_idx]:.4f}")

    print(f"\nGenerating plot...")
    plot_results(k_values, custom_scores, sklearn_scores, weighted_scores, "R²")


if __name__ == "__main__":
    main()
