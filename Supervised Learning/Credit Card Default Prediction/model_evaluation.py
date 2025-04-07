import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# Load preprocessed data
X_test = pd.read_csv('X_test_preprocessed.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

# Load the best tuned model
best_model = joblib.load('best_model_tuned.pkl')

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print(f'Tuned Model Accuracy: {accuracy}')
print(f'Tuned Model ROC-AUC: {roc_auc}')
print(classification_report(y_test, y_pred))