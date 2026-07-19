import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import pairwise_distances
import warnings
import matplotlib.pyplot as plt

#A1
df = pd.read_excel('Lab Session Data.xlsx', sheet_name='Purchase data')

df = df.iloc[:, :5]
X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
rank = np.linalg.matrix_rank(X)


xinv = np.linalg.pinv(X)
y = df['Payment (Rs)'].values.reshape(-1, 1)

#A2

cost = xinv @ y
print(f"Rank = {rank}")
print(df)
print(f"Pseudoinverse of the matrix X: {xinv}")
print(f"Cost vector: {cost}")

for i in range(len(y)):
    if(y[i]<200):
        print(f"Customer {i+1} is Poor")
    else:
        print(f"Customer {i+1} is Rich")
#A3


df= pd.read_excel('Lab Session Data.xlsx', sheet_name='IRCTC Stock Price')


df = df.iloc[:, :9]
print((df))
D = df['Price'].values.reshape(-1, 1)
sum2 = 0 
for i in range(len(D)):
    sum2 += D[i]
mean = sum2/len(D)
print(f"Mean of the stock price = {mean}")
var = np.var(D)
variance = sum((x - mean) ** 2 for x in D) / len(D)
print(f"Variance of the stock price = {var}")
print(f"Variance of the stock price = {variance}")

print(" First 5 rows of the data : ")
print(df.head())

print("Wednesday analysis : ")

wed_data = df[df['Day']=='Wed']

wed_mean = wed_data['Price'].mean()

population_mean = df['Price'].mean()

print(f"Total records = {len(df)}")
print(f"Number of wednesday records : {len(wed_data)}")

print(f"Wednesday mean : {wed_mean}")
print(f"Population Mean : {population_mean}")


april_data = df[df['Month'] == 'Apr']


april_mean = april_data['Price'].mean()
print(f"\nNumber of April records: {len(april_data)}")
    
print(f"April Sample Mean (NumPy):     {april_mean}")
print(f"Population Mean (All data):    {population_mean}")


print("5. CONDITIONAL PROBABILITY")


if 'Chg%' in df.columns and 'Day' in df.columns:
   
    total_records = len(df)
    wed_total = len(df[df['Day'] == 'Wed'])
    
    p_wednesday = wed_total / total_records
  
    profit_and_wed = len(df[(df['Chg%'] > 0) & (df['Day'] == 'Wed')])
    p_profit_and_wed = profit_and_wed / total_records
    
    if wed_total > 0:
        conditional_prob = profit_and_wed / wed_total
        
        print(f"\nTotal records: {total_records}")
        print(f"Wednesday records: {wed_total}")
        print(f"Profitable Wednesday records: {profit_and_wed}")
        
        print(f"\n--- Probability Calculations ---")
        print(f"P(Wednesday) = {wed_total}/{total_records} = {p_wednesday:.2%}")
        print(f"P(Profit ∩ Wednesday) = {profit_and_wed}/{total_records} = {p_profit_and_wed:.2%}")
        print(f"\nP(Profit | Wednesday) = P(Profit ∩ Wednesday) / P(Wednesday)")
        print(f"P(Profit | Wednesday) = {profit_and_wed}/{wed_total} = {conditional_prob:.2%}")
        
        print(f"\nInterpretation: If today is Wednesday, there's a {conditional_prob:.2%} chance of making a profit.")
        
        p_profit = len(df[df['Chg%'] > 0]) / len(df['Chg%'].dropna())
        print(f"\nP(Profit) = {p_profit:.2%}")
        
    else:
        print("No Wednesday data found")

print("6. SCATTER PLOT - Chg% vs Day of Week")


if 'Chg%' in df.columns and 'Day' in df.columns:
  
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    day_map = {day: i for i, day in enumerate(day_order)}
    
    df['Day_Num'] = df['Day'].map(day_map)
    
    plot_data = df.dropna(subset=['Chg%', 'Day_Num'])
    
    if len(plot_data) > 0:
      
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
       
        ax1.scatter(plot_data['Day_Num'], plot_data['Chg%'], 
                   alpha=0.6, color='blue', s=50)
        
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, 
                   label='Zero line (Profit/Loss boundary)')
        
        mean_chg = plot_data['Chg%'].mean()
        ax1.axhline(y=mean_chg, color='green', linestyle='-', linewidth=1, alpha=0.5,
                   label=f'Mean Chg% = {mean_chg:.2f}%')
        
       
        ax1.set_xticks(range(len(day_order)))
        ax1.set_xticklabels(day_order)
        
        ax1.set_xlabel('Day of Week')
        ax1.set_ylabel('Chg% (Price Change %)')
        ax1.set_title('Scatter Plot: Chg% by Day of Week')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        box_data = [plot_data[plot_data['Day'] == day]['Chg%'].dropna() for day in day_order]
        bp = ax2.boxplot(box_data, labels=day_order, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightgray']):
            patch.set_facecolor(color)
        
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, 
                   label='Zero line (Profit/Loss boundary)')
        ax2.axhline(y=mean_chg, color='green', linestyle='-', linewidth=1, alpha=0.5,
                   label=f'Mean Chg% = {mean_chg:.2f}%')
        
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Chg% (Price Change %)')
        ax2.set_title('Box Plot: Chg% Distribution by Day')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("\nSummary Statistics by Day:")
        print("-" * 70)
        day_stats = plot_data.groupby('Day')['Chg%'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(day_stats)
        
        print("\nProfit Probability by Day:")
        print("-" * 50)
        for day in day_order:
            day_data = plot_data[plot_data['Day'] == day]
            if len(day_data) > 0:
                profit_prob = len(day_data[day_data['Chg%'] > 0]) / len(day_data)
                loss_prob = len(day_data[day_data['Chg%'] < 0]) / len(day_data)
                print(f"{day:<5}: Profit: {profit_prob:.2%}  |  Loss: {loss_prob:.2%}  |  Count: {len(day_data)}")
    else:
        print("No data available for plotting")

print("SUMMARY OF OBSERVATIONS")



print(f"\nActual Summary Values:")
print(f"Population Mean Price: {population_mean:,.2f}")
print(f"Wednesday Mean Price: {wed_mean:,.2f}")
print(f"April Mean Price: {april_mean:,.2f}")
print(f"Overall Profit Probability: {len(df[df['Chg%'] > 0]) / len(df['Chg%'].dropna()):.2%}")
print(f"Wednesday Profit Probability: {profit_prob:.2%}")
print(f"P(Profit | Wednesday): {conditional_prob:.2%}")


warnings.filterwarnings('ignore')

# A4. DATA EXPLORATION

print("="*80)
print("A4. DATA EXPLORATION - Thyroid Dataset")
print("="*80)

df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

print("\n1. Dataset Overview:")
print("-"*50)
print(f"Shape: {df.shape} (rows, columns)")
print(f"\nColumn names:")
print(df.columns.tolist())

print("\n\n2. First 5 rows:")
print(df.head())

print("\n\n3. Data Types and Info:")
print("-"*50)
print(df.info())

print("\n\n4. Attribute Study:")
print("-"*50)

for col in df.columns:
    print(f"\n--- {col} ---")
    print(f"Data Type: {df[col].dtype}")
    print(f"Unique Values: {df[col].nunique()}")
    
    if df[col].dtype == 'object' or df[col].nunique() < 20:
        
        print(f"Value Counts:\n{df[col].value_counts().head()}")
        
        if df[col].nunique() <= 2:
            print("Encoding: Binary (Label Encoding)")
        elif df[col].dtype == 'object':
           
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 2:
                print("Encoding: One-Hot Encoding (Nominal)")
            else:
                print("Encoding: Label Encoding (Ordinal)")
        else:
            print("Encoding: Label Encoding")
    else:
      
        print(f"Data Range: {df[col].min():.2f} to {df[col].max():.2f}")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Std Dev: {df[col].std():.2f}")
        print(f"Missing Values: {df[col].isnull().sum()}")


print("\n\n5. Missing Values Analysis:")
print("-"*50)

missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Values': missing_data,
    'Percentage': missing_percent
})
print(missing_df[missing_df['Missing Values'] > 0])


print("\n\n6. Outlier Detection (Numeric Variables):")
print("-"*50)

numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    
    if outlier_count > 0:
        print(f"\n{col}:")
        print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        print(f"  Outliers: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
        print(f"  Outlier values: {outliers[col].head().tolist()}")



print("\n\n7. Numeric Variables Statistics:")
print("-"*50)

numeric_stats = df[numeric_cols].describe()
print(numeric_stats)

# A5. JACCARD & SIMILARITY COEFFICIENT


print("\n" + "="*80)
print("A5. JACCARD AND SIMILARITY COEFFICIENT")
print("="*80)

binary_cols = []
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    if set(unique_vals).issubset({0, 1}) or set(unique_vals).issubset({0.0, 1.0}):
        binary_cols.append(col)

print(f"\nBinary columns found: {binary_cols}")

if len(binary_cols) >= 2:
    vec1 = df.iloc[0][binary_cols].values
    vec2 = df.iloc[1][binary_cols].values
    
    f11 = np.sum((vec1 == 1) & (vec2 == 1))
    f10 = np.sum((vec1 == 1) & (vec2 == 0))
    f01 = np.sum((vec1 == 0) & (vec2 == 1))
    f00 = np.sum((vec1 == 0) & (vec2 == 0))
    
    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) > 0 else 0
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    
    print(f"\nFirst Observation Vector (Binary): {vec1}")
    print(f"Second Observation Vector (Binary): {vec2}")
    print(f"\nContingency Table:")
    print(f"f11 (1,1): {f11}")
    print(f"f10 (1,0): {f10}")
    print(f"f01 (0,1): {f01}")
    print(f"f00 (0,0): {f00}")
    print(f"\nJaccard Coefficient (JC): {jc:.4f}")
    print(f"Simple Matching Coefficient (SMC): {smc:.4f}")
    print(f"\nInterpretation:")
    print(f"JC = {jc:.2%} - Only considers positive matches")
    print(f"SMC = {smc:.2%} - Considers both positive and negative matches")
    
    if jc > smc:
        print("JC > SMC: Positive matches are more important than negative matches")
    else:
        print("SMC > JC: Negative matches also contribute to similarity")
else:
    print("Not enough binary columns found.")

# A6. COSINE SIMILARITY


print("\n" + "="*80)
print("A6. COSINE SIMILARITY")
print("="*80)

vec1_complete = df.iloc[0].values.astype(float)
vec2_complete = df.iloc[1].values.astype(float)

# Handle NaN values
vec1_complete = np.nan_to_num(vec1_complete)
vec2_complete = np.nan_to_num(vec2_complete)

# Calculate Cosine Similarity
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

cos_sim = cosine_similarity(vec1_complete, vec2_complete)

print(f"\nComplete Vector 1: {vec1_complete[:10]}... (showing first 10 values)")
print(f"Complete Vector 2: {vec2_complete[:10]}... (showing first 10 values)")
print(f"\nCosine Similarity: {cos_sim:.4f}")
print(f"Interpretation: Cosine similarity of {cos_sim:.2%} indicates the angle between the vectors")

# A7. HEATMAP PLOT


print("\n" + "="*80)
print("A7. HEATMAP PLOT - Similarity Matrix")
print("="*80)

n_samples = min(20, len(df))
sample_data = df.iloc[:n_samples]

def calculate_similarities(data, method='jaccard'):
    if method == 'jaccard':
       
        binary_data = data[binary_cols].values if len(binary_cols) > 0 else data.values
        
        binary_data = np.nan_to_num(binary_data)
      
        similarity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    similarity_matrix[i, j] = 1
                else:
                    v1 = binary_data[i]
                    v2 = binary_data[j]
                    f11 = np.sum((v1 == 1) & (v2 == 1))
                    f10 = np.sum((v1 == 1) & (v2 == 0))
                    f01 = np.sum((v1 == 0) & (v2 == 1))
                    similarity_matrix[i, j] = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) > 0 else 0
        return similarity_matrix
    
    elif method == 'smc':
       
        binary_data = data[binary_cols].values if len(binary_cols) > 0 else data.values
        binary_data = np.nan_to_num(binary_data)
        similarity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    similarity_matrix[i, j] = 1
                else:
                    v1 = binary_data[i]
                    v2 = binary_data[j]
                    f11 = np.sum((v1 == 1) & (v2 == 1))
                    f10 = np.sum((v1 == 1) & (v2 == 0))
                    f01 = np.sum((v1 == 0) & (v2 == 1))
                    f00 = np.sum((v1 == 0) & (v2 == 0))
                    similarity_matrix[i, j] = (f11 + f00) / (f00 + f01 + f10 + f11)
        return similarity_matrix
    
    elif method == 'cosine':
       
        full_data = data.values.astype(float)
        full_data = np.nan_to_num(full_data)
        similarity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    similarity_matrix[i, j] = 1
                else:
                    similarity_matrix[i, j] = cosine_similarity(full_data[i], full_data[j])
        return similarity_matrix

jc_matrix = calculate_similarities(sample_data, 'jaccard')
smc_matrix = calculate_similarities(sample_data, 'smc')
cos_matrix = calculate_similarities(sample_data, 'cosine')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.heatmap(jc_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0])
axes[0].set_title('Jaccard Similarity Matrix\n(First 20 Observations)')
axes[0].set_xlabel('Observation Index')
axes[0].set_ylabel('Observation Index')

sns.heatmap(smc_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1])
axes[1].set_title('SMC Similarity Matrix\n(First 20 Observations)')
axes[1].set_xlabel('Observation Index')
axes[1].set_ylabel('Observation Index')

sns.heatmap(cos_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[2])
axes[2].set_title('Cosine Similarity Matrix\n(First 20 Observations)')
axes[2].set_xlabel('Observation Index')
axes[2].set_ylabel('Observation Index')

plt.tight_layout()
plt.show()

# A8. DATA IMPUTATION


print("\n" + "="*80)
print("A8. DATA IMPUTATION")
print("="*80)

df_imputed = df.copy()

for col in df_imputed.columns:
    missing_count = df_imputed[col].isnull().sum()
    if missing_count > 0:
        print(f"\nImputing {col} ({missing_count} missing values):")
        
        if df_imputed[col].dtype == 'object':
           
            mode_val = df_imputed[col].mode()[0]
            df_imputed[col].fillna(mode_val, inplace=True)
            print(f"  Used Mode: {mode_val}")
        else:
         
            Q1 = df_imputed[col].quantile(0.25)
            Q3 = df_imputed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            has_outliers = ((df_imputed[col] < lower_bound) | (df_imputed[col] > upper_bound)).any()
            
            if has_outliers:
               
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val, inplace=True)
                print(f"  Used Median (outliers present): {median_val:.2f}")
            else:
              
                mean_val = df_imputed[col].mean()
                df_imputed[col].fillna(mean_val, inplace=True)
                print(f"  Used Mean (no outliers): {mean_val:.2f}")

print("\nMissing values after imputation:")
print(df_imputed.isnull().sum())

# A9. DATA NORMALIZATION


print("\n" + "="*80)
print("A9. DATA NORMALIZATION / SCALING")
print("="*80)

numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns

print(f"\nNumeric columns to normalize: {numeric_cols.tolist()}")

df_normalized = df_imputed.copy()

scaler_standard = StandardScaler()
df_standard = df_normalized.copy()
df_standard[numeric_cols] = scaler_standard.fit_transform(df_standard[numeric_cols])

scaler_minmax = MinMaxScaler()
df_minmax = df_normalized.copy()
df_minmax[numeric_cols] = scaler_minmax.fit_transform(df_minmax[numeric_cols])

scaler_robust = RobustScaler()
df_robust = df_normalized.copy()
df_robust[numeric_cols] = scaler_robust.fit_transform(df_robust[numeric_cols])

print("\nComparison of normalization methods (first 5 rows):")
print("\nOriginal Data:")
print(df_normalized[numeric_cols].head())

print("\nStandardized (Z-score):")
print(df_standard[numeric_cols].head())

print("\nMin-Max Scaled (0-1):")
print(df_minmax[numeric_cols].head())

print("\nRobust Scaled (Median/IQR):")
print(df_robust[numeric_cols].head())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, col in enumerate(numeric_cols[:4]):  # Plot first 4 numeric columns
    ax = axes[i // 2, i % 2]
    ax.hist(df_normalized[col].dropna(), bins=20, alpha=0.5, label='Original', color='blue')
    ax.hist(df_standard[col].dropna(), bins=20, alpha=0.5, label='Standardized', color='green')
    ax.hist(df_minmax[col].dropna(), bins=20, alpha=0.5, label='Min-Max', color='red')
    ax.set_title(f'{col} Distribution')
    ax.legend()

plt.tight_layout()
plt.show()


print("SUMMARY OF ANALYSIS")

print("\nRecommendations:")
print("-"*50)
print("1. For categorical attributes: Use One-Hot Encoding for nominal, Label Encoding for ordinal")
print("2. For missing values: Use mean/median/mode based on data characteristics")
print("3. For similarity: Jaccard for binary data, Cosine for continuous data")
print("4. For normalization: StandardScaler for normally distributed data, MinMaxScaler for bounded data, RobustScaler for data with outliers")