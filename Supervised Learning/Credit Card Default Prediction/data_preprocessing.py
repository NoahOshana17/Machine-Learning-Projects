from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

# Custom transformer for feature engineering
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['AVG_PAY'] = X[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)
        X['TOTAL_BILL_AMT'] = X[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)
        X = X.drop(columns=['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'])
        return X

# Fetch dataset
default_of_credit_card_clients = fetch_ucirepo(id=350)

# Data (as pandas dataframes)
X = default_of_credit_card_clients.data.features 
y = default_of_credit_card_clients.data.targets 

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

# Split data into features and target
X = df[feature_names]
y = df[target_name]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
    'AVG_PAY', 'TOTAL_BILL_AMT'
]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Create a pipeline that includes feature engineering and preprocessing
pipeline = Pipeline(steps=[
    ('feature_engineering', FeatureEngineering()),
    ('preprocessor', preprocessor)
])

# Preprocess the data
X_train_preprocessed = pipeline.fit_transform(X_train)
X_test_preprocessed = pipeline.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
print(len(X_train_resampled), len(y_train_resampled))

# Save the preprocessed and resampled data for model training
pd.DataFrame(X_train_resampled, columns=pipeline.named_steps['preprocessor'].get_feature_names_out()).to_csv('X_train_preprocessed.csv', index=False)
pd.DataFrame(X_test_preprocessed, columns=pipeline.named_steps['preprocessor'].get_feature_names_out()).to_csv('X_test_preprocessed.csv', index=False)
pd.DataFrame(y_train_resampled).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)