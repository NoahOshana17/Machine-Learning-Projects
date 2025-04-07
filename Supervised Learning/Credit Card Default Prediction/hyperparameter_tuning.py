import pandas as pd
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Load preprocessed data
X_train = pd.read_csv('X_train_preprocessed.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# # Define parameter grid for XGBoost
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'min_child_weight': [1, 3, 5],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }

# # Perform grid search
# grid_search = GridSearchCV(XGBClassifier(eval_metric='logloss'), param_grid, cv=5, scoring='roc_auc')
# grid_search.fit(X_train, y_train)

# # Best parameters and score
# print(f'Best Parameters: {grid_search.best_params_}')
# print(f'Best ROC-AUC: {grid_search.best_score_}')

# # Save the best model
# best_model = grid_search.best_estimator_
# joblib.dump(best_model, 'best_model_tuned.pkl')

# Define parameter search space for XGBoost
param_space = {
    'n_estimators': (100, 300),
    'learning_rate': (0.01, 0.2, 'log-uniform'),
    'max_depth': (3, 7),
    'min_child_weight': (1, 5),
    'subsample': (0.8, 1.0),
    'colsample_bytree': (0.8, 1.0)
}

# Perform Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=XGBClassifier(eval_metric='logloss'),
    search_spaces=param_space,
    n_iter=50,  # Number of iterations
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
bayes_search.fit(X_train, y_train)

# Best parameters and score
print(f'Best Parameters: {bayes_search.best_params_}')
print(f'Best ROC-AUC: {bayes_search.best_score_}')

# Save the best model
best_model = bayes_search.best_estimator_
joblib.dump(best_model, 'best_model_tuned.pkl')

