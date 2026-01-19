import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.compose import ColumnTransformer

df = pd.read_csv('diabetes.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 7 rows of the dataset:")
print(df.head(7))

X = df.drop('Outcome', axis=1)
y =df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = X.columns

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features)
])

svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', SVC(kernel='rbf', probability=True, random_state=42, C=10, gamma=0.01))
])

svm_pipeline.fit(X_train, y_train)

y_pred = svm_pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

with open('diabetes_svc_model.pkl', 'wb') as f:
    pickle.dump(svm_pipeline, f)

print('SVM pipeline saved...')    

