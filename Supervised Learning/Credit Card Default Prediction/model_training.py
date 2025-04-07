import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
from xgboost import XGBClassifier

# Load preprocessed data
X_train = pd.read_csv('X_train_preprocessed.csv')
X_test = pd.read_csv('X_test_preprocessed.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Define models to train
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

# Initialize variables to keep track of the best model
best_model = None
best_score = 0
best_model_name = ""

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}')
    print(f'{name} ROC-AUC: {roc_auc}')
    print(classification_report(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    mean_cv_score = cv_scores.mean()
    print(f'{name} Cross-Validation ROC-AUC: {mean_cv_score}')

    # Check if this model is the best so far
    if mean_cv_score > best_score:
        best_score = mean_cv_score
        best_model = model
        best_model_name = name

# Save the best model
print(f'Best Model: {best_model_name} with ROC-AUC: {best_score}')
joblib.dump(best_model, 'best_model.pkl')