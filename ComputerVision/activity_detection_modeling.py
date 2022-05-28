"""## Import Requirements"""

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import eli5
from glob import glob
import pickle
# from imblearn.over_sampling import SMOTE

"""## Data Preperation"""

df_model = pd.read_excel('working-detection.xlsx', engine='openpyxl')
X = df_model.drop(columns='label')
y = df_model['label']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# balance the classes in the training set (result: no gain)
# oversample = SMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train)

# balance sides in the training set (mirror images)
X_mirrored = X_train.copy()
X_mirrored = X_mirrored[['RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW', 
       'RIGHT_HIP', 'LEFT_HIP', 'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE']]
X_mirrored.columns = X_train.columns
X_train = pd.concat([X_train, X_mirrored])
y_train = pd.concat([y_train, y_train])

"""## Try the first model"""

model = LGBMClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('First Results:', pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)), sep='\n')

"""## Model Selection"""

pipeline = Pipeline(steps=[("scaler", MinMaxScaler()), ("classifier", LGBMClassifier())])

params = [
    {
      'scaler': [StandardScaler(), MinMaxScaler()],
      'classifier': [LogisticRegression()],
      "classifier__C": [0.1, 1.0, 10.0, 100.0],
    },
    {
      'scaler': [StandardScaler(), MinMaxScaler()],
      'classifier': [RandomForestClassifier()],
      'classifier__max_depth': np.arange(1, 22, 2),
      'classifier__n_estimators': np.arange(10, 500, 50),
    },
    {
      'scaler': [StandardScaler(), MinMaxScaler()],
      'classifier': [LGBMClassifier()],
      'classifier__max_depth': np.arange(1, 22, 2),
      'classifier__num_leaves': np.arange(2, 103, 5),
      'classifier__n_estimators': np.arange(10, 311, 50),
      'classifier__learning_rate': np.arange(0.01, 1.502, 0.05)
    },
]

print('Tuning the model...')
search = RandomizedSearchCV(pipeline, params, n_iter=1000, cv=10, random_state=42)
search.fit(X_train, y_train)

print('Best Estimator:', search.best_estimator_)
print('Best Score:', search.best_score_)

results = pd.DataFrame(search.cv_results_)

results_ = results.loc[results['param_classifier'].apply(lambda x: type(x)==LGBMClassifier)]
for col in results_.columns:
  if 'param_classifier__' in col:
    sns.set( rc = {'figure.figsize' : ( 12, 7 ), 
                  'axes.labelsize' : 12 })
    plt.title(col.split('__')[1], size = 16)
    g = sns.scatterplot(results_[col], results_['mean_test_score'])
    # g.set(ylim=(0.515, 0.525))
    plt.show()

"""## Model Evaluation"""

model = Pipeline(steps=[('scaler', StandardScaler()),
                ('classifier',
                 LGBMClassifier(learning_rate=1.4600000000000002, max_depth=15,
                                n_estimators=310, num_leaves=82))])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Results of the Tuned Model:', pd.DataFrame(classification_report(y_test, y_pred>0.5, output_dict=True)), sep='\n')

# save the model
model_path = 'activity_detection_model.pkl'
model_file = open(model_path, 'wb')
pickle.dump(model, model_file)
model_file.close()

# TODO: handle sides issue
## option 1: balance sides of the photos
## option 2: pick side by visibility 

# TODO: add object at hand detection

"""## Model Explanation"""

feature_weights = eli5.explain_weights_df(model, feature_names=X_train.columns)
print('Results of the Tuned Model:', feature_weights, sep='\n')

"""## For Later"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Initialize the model
model = keras.Sequential()
# Add layers
model.add(layers.Dense(5, activation='softmax', input_dim=10))
model.add(layers.Dense(2, activation='relu'))
# model.add(layers.Dense(3, activation='softmax'))
model.add(layers.Dense(1, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)

pd.DataFrame(classification_report(y_test, y_pred>0.5, output_dict=True))

