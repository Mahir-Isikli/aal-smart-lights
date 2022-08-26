# ## Import Requirements
import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import eli5
import shap
from glob import glob
import time
import gc
import pickle
import tensorflow as tf
import tensorflow_hub as hub

# ## Data Preperation
df_model = pd.read_excel(
    'working-detection-final.xlsx', engine='openpyxl')
X = df_model.drop(columns='label')
y = df_model['label']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# balance sides in the training set (mirror images)
X_mirrored = X_train.copy()
mirrored_cols = ['RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW',
                 'RIGHT_HIP', 'LEFT_HIP', 'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE']
mirrored_cols = [[c, c+'_visibility'] for c in mirrored_cols]
mirrored_cols = [item for sublist in mirrored_cols for item in sublist]
mirrored_cols = mirrored_cols + ['looks_at_laptop', 'looks_at_keyboard',
                                 'looks_at_cellphone', 'hand_at_laptop', 'hand_at_keyboard', 'hand_at_cellphone']
X_mirrored = X_mirrored[mirrored_cols]
X_mirrored.columns = X_train.columns
X_train = pd.concat([X_train, X_mirrored])
y_train = pd.concat([y_train, y_train])


# ## Try the first model
model = LGBMClassifier()
print('Selection score of the deafult hyperparams:', np.mean(
    cross_validate(model, X_train, y_train, cv=10)['test_score']))


# ## Model Selection

pipeline = Pipeline(steps=[("scaler", MinMaxScaler()),
                    ("classifier", LGBMClassifier())])

params = [
    {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'classifier': [LogisticRegression()],
        "classifier__C": [0.1, 1.0, 10.0, 100.0],
    },
    {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'classifier': [RandomForestClassifier()],
        'classifier__max_depth': np.arange(3, 22, 2),
        'classifier__n_estimators': np.arange(10, 311, 50),
    },
    {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'classifier': [LGBMClassifier()],
        'classifier__max_depth': np.arange(3, 17, 2),
        'classifier__num_leaves': np.arange(2, 203, 5),
        'classifier__n_estimators': np.arange(10, 311, 50),
        'classifier__learning_rate': np.arange(0.01, 1.502, 0.05)
    },
]

print('Tuning the model...')
search = RandomizedSearchCV(
    pipeline, params, n_iter=500, cv=10, random_state=42)
search.fit(X_train, y_train)

print('Best Estimator:', search.best_estimator_)
print('Best Score:', search.best_score_)


# ## Model Evaluation
model = Pipeline(steps=[('scaler', MinMaxScaler()),
                        ('classifier',
                 LGBMClassifier(learning_rate=0.060000000000000005,
                                max_depth=11, n_estimators=210,
                                num_leaves=32))])


# uncomment to load a pretrained model
# model = pickle.load(open('activity_detection_v2.pkl', 'rb'))

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Results of the Tuned Model:', pd.DataFrame(
    classification_report(y_test, y_pred > 0.5, output_dict=True)), sep='\n')

# save the model
model_path = 'activity_detection_vXXX.pkl'
model_file = open(model_path, 'wb')
pickle.dump(model, model_file)
model_file.close()


# ## Model Explanation

# ELI5
feature_weights = eli5.explain_weights_df(model, feature_names=X_train.columns)
print('Feature weights of the Tuned Model:', feature_weights, sep='\n')

# SHAP
# Fits the explainer
explainer = shap.Explainer(model.predict, X_test)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test)

print('Mean SHAP values for class 0:')
print(pd.DataFrame(shap_values.values[y_test==0].mean(axis=0).reshape((1, 26)), columns = X_train.columns).T)

print('Mean SHAP values for class 1:')
print(pd.DataFrame(shap_values.values[y_test==1].mean(axis=0).reshape((1, 26)), columns = X_train.columns).T)

