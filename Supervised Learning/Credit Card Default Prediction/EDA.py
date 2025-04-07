from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# fetch dataset 
default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
# data (as pandas dataframes) 
X = default_of_credit_card_clients.data.features 
y = default_of_credit_card_clients.data.targets 
  
# metadata 
print(default_of_credit_card_clients.metadata) 

# Rename columns based on metadata
feature_names = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]
target_name = ['default.payment.next.month']

# Rename the columns of X and y
X.columns = feature_names
y.columns = target_name

# Join X and y to a single pandas DataFrame
df = pd.concat([X, y], axis=1)
print(df.head(10))

# Display basic information about the DataFrame
print("Basic Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x=target_name[0], data=df)
plt.title('Distribution of Target Variable')
plt.show()

# Visualize correlations between features
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# # Distribution of numerical features
# plt.figure(figsize=(20, 15))
# for i, col in enumerate(feature_names):
#     plt.subplot(6, 4, i + 1)
#     sns.histplot(df[col], kde=True)
#     plt.title(f'Distribution of {col}')
# plt.tight_layout()
# plt.show()

# # Box plots to identify outliers
# plt.figure(figsize=(20, 15))
# for i, col in enumerate(feature_names):
#     plt.subplot(6, 4, i + 1)
#     sns.boxplot(y=df[col])
#     plt.title(f'Box plot of {col}')
# plt.tight_layout()
# plt.show()

# # Pair plots to visualize relationships between pairs of features
# sns.pairplot(df[feature_names + target_name], diag_kind='kde', hue=target_name[0])
# plt.show()

# # Analyze the relationship between features and the target variable
# plt.figure(figsize=(20, 15))
# for i, col in enumerate(feature_names):
#     plt.subplot(6, 4, i + 1)
#     sns.boxplot(x=target_name[0], y=col, data=df)
#     plt.title(f'{col} vs {target_name[0]}')
# plt.tight_layout()
# plt.show()

# Feature importance using a simple model
from sklearn.ensemble import RandomForestClassifier

X = df[feature_names]
y = df[target_name[0]]

model = RandomForestClassifier()
model.fit(X, y)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()