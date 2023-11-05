import numpy as np
import pandas as pd
import io
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
# Import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
#Import Kfold
from sklearn.model_selection import KFold, StratifiedKFold
#import SMOTE
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from google.colab import files
uploaded = files.upload()

data = pd.read_csv(io.BytesIO(uploaded['WA_Fn-UseC_-Telco-Customer-Churn.csv']))

data

data.info()

data.isna().sum()

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dtypes

data.dropna(inplace=True)

data.drop('customerID', axis=1, inplace=True)

# Converting Categorical columns to dummy variables
data['Churn'].replace(to_replace='Yes', value=1, inplace=True)
data['Churn'].replace(to_replace='No',  value=0, inplace=True)

data_dummies = pd.get_dummies(data)

data_dummies

# Plotting Correlation of "Churn" with other variables:
plt.figure(figsize=(15,8))
cmap = plt.get_cmap('RdYlBu')
colors = cmap(data_dummies.corr()['Churn'].values)
data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar', color=colors)

X = data_dummies.drop('Churn', axis=1)
y = data_dummies['Churn']

features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

y_train.value_counts()

print("Distribution of target variable in training set before applying SMOTE: ", y_train.value_counts(), sep='\n')

sm = SMOTE(random_state=123)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("\nDistribution of target variable in training set after applying SMOTE: ", y_train_sm.value_counts(), sep='\n')

print("Distribution of target variable in testing set before applying SMOTE: ", y_test.value_counts(), sep='\n')

sm = SMOTE(random_state=123)
X_test_sm, y_test_sm = sm.fit_resample(X_test, y_test)

print("\nDistribution of target variable in testing set after applying SMOTE: ", y_test_sm.value_counts(), sep='\n')

"""**Model Building**

## 1. Logistic Regression

### Training on Imbalanced Training Data and Testing on Imbalanced Testing Data
"""

log_reg = LogisticRegression()
result = log_reg.fit(X_train, y_train)
y_preds = log_reg.predict(X_test)
print(classification_report(y_test, y_preds))

"""### Training on Balanced Training Data and Testing on Balanced Testing Data"""

log_reg = LogisticRegression()
result = log_reg.fit(X_train_sm, y_train_sm)
y_preds = log_reg.predict(X_test_sm)
print(classification_report(y_test_sm, y_preds))

"""### Training on Balanced Training Data and Testing on Imbalanced Testing Data"""

log_reg = LogisticRegression()
result = log_reg.fit(X_train_sm, y_train_sm)
y_preds = log_reg.predict(X_test)
print(classification_report(y_test, y_preds))

# Do gridsearch across 10 folds for Logistic Regression
log_reg = LogisticRegression()
param_grid = {
            'C': np.logspace(-4, 4, 50),
            'penalty': ['l2', 'none'],
            'solver': ['lbfgs', 'sag', 'saga']
            }
log_reg_cv = GridSearchCV(log_reg, param_grid, cv=10, scoring = 'recall', n_jobs=-1, verbose=1)
log_reg_cv.fit(X_train, y_train)
print("Tuned Logistic Regression Parameters: {}".format(log_reg_cv.best_params_))
print("Best score is {}\n\n".format(log_reg_cv.best_score_))

y_preds = log_reg_cv.best_estimator_.predict(X_test)
print(classification_report(y_test, y_preds))

# Doing gridsearch across 10 folds for Logistic Regression with SMOTE
log_reg = LogisticRegression()
param_grid = {
            'C': np.logspace(-4, 4, 50),
            'penalty': ['l2', 'none'],
            'solver': ['lbfgs', 'sag', 'saga']
            }
log_reg_cv = GridSearchCV(log_reg, param_grid, cv=10, scoring = 'recall', n_jobs=-1, verbose=1)
log_reg_cv.fit(X_train_sm, y_train_sm)
print("Tuned Logistic Regression Parameters: {}".format(log_reg_cv.best_params_))
print("Best score is {}\n\n".format(log_reg_cv.best_score_))

y_preds = log_reg_cv.best_estimator_.predict(X_test_sm)
print(classification_report(y_test_sm, y_preds))

y_preds = log_reg_cv.best_estimator_.predict(X_test)
print(classification_report(y_test, y_preds))

"""## 2. Random Forest"""

rf = RandomForestClassifier()
result = rf.fit(X_train, y_train)
y_preds = rf.predict(X_test)
print(classification_report(y_test, y_preds))

rf = RandomForestClassifier()
result = rf.fit(X_train_sm, y_train_sm)
y_preds = rf.predict(X_test_sm)
print(classification_report(y_test_sm, y_preds))

y_preds = rf.predict(X_test)
print(classification_report(y_test, y_preds))

# Tuning Hyperparameters of RF
rf = RandomForestClassifier()
param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'criterion' :['gini', 'entropy']
            }
rf_cv = GridSearchCV(rf, param_grid, cv=10, scoring = 'recall', n_jobs=-1, verbose=1)
rf_cv.fit(X_train, y_train)
print("Tuned Random Forest Parameters: {}".format(rf_cv.best_params_))
print("Best score is {}\n\n".format(rf_cv.best_score_))

y_preds = rf_cv.best_estimator_.predict(X_test)
print(classification_report(y_test, y_preds))

# Tuning Hyperparameters of RF
rf = RandomForestClassifier()
param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'criterion' :['gini', 'entropy']
            }
rf_cv = GridSearchCV(rf, param_grid, cv=10, scoring = 'recall', n_jobs=-1, verbose=1)
rf_cv.fit(X_train_sm, y_train_sm)
print("Tuned Random Forest Parameters: {}".format(rf_cv.best_params_))
print("Best score is {}\n\n".format(rf_cv.best_score_))

y_preds = rf_cv.best_estimator_.predict(X_test_sm)
print(classification_report(y_test_sm, y_preds))

y_preds = rf_cv.best_estimator_.predict(X_test)
print(classification_report(y_test, y_preds))

"""## 3. XGBoost"""

xgb = XGBClassifier()
result = xgb.fit(X_train, y_train)
y_preds = xgb.predict(X_test)
print(classification_report(y_test, y_preds))

# Tuning Hyperparameters of XGB
xgb = XGBClassifier()

param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
            }
xgb_cv = GridSearchCV(xgb, param_grid, cv=10, scoring = 'recall', n_jobs=-1, verbose=1)
xgb_cv.fit(X_train, y_train)
print("Tuned XGB Parameters: {}".format(xgb_cv.best_params_))
print("Best score is {}\n\n".format(xgb_cv.best_score_))

y_preds = xgb_cv.best_estimator_.predict(X_test)
print(classification_report(y_test, y_preds))

# Tuning Hyperparameters of XGB with SMOTE
xgb = XGBClassifier()

param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
            }
xgb_cv = GridSearchCV(xgb, param_grid, cv=10, scoring = 'recall', n_jobs=-1, verbose=1)
xgb_cv.fit(X_train_sm, y_train_sm)
print("Tuned XGB Parameters: {}".format(xgb_cv.best_params_))
print("Best score is {}\n\n".format(xgb_cv.best_score_))

y_preds = xgb_cv.best_estimator_.predict(X_test_sm)
print(classification_report(y_test_sm, y_preds))

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):



    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14
labels = ['No Churn', 'Churn']
cm = [[1244, 240], [199, 427]]

"""## Refining the model"""

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Implementing StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=123)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Distribution of target variable in training set: ", y_train.value_counts())

# Applying SMOTE only on training data
sm = SMOTE(random_state=123)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print("Distribution of target variable in training set after SMOTE: ", y_train_sm.value_counts())

# Expanding the hyperparameters for XGB
xgb = XGBClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 5, 7],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Using StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
xgb_cv = GridSearchCV(xgb, param_grid, cv=skf, scoring='recall', n_jobs=-1, verbose=1)
xgb_cv.fit(X_train_sm, y_train_sm)

print("Tuned XGB Parameters: {}".format(xgb_cv.best_params_))
print("Best score is {}\n".format(xgb_cv.best_score_))

y_preds = xgb_cv.best_estimator_.predict(X_test)
print(classification_report(y_test, y_preds))

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Implementing StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=123)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Applying SMOTE only on training data
sm = SMOTE(random_state=123)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Feature Selection using Recursive Feature Elimination
rfe = RFE(estimator=XGBClassifier(), n_features_to_select=15)
X_train_sm_rfe = rfe.fit_transform(X_train_sm, y_train_sm)
X_test_rfe = rfe.transform(X_test)

# Model Ensembling
xgb = XGBClassifier()
rf = RandomForestClassifier()

voting_classifier = VotingClassifier(estimators=[
    ('xgb', xgb), ('rf', rf)
], voting='soft')

param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [4, 5],
    'xgb__learning_rate': [0.05, 0.1],
    'xgb__gamma': [0, 0.1],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0],
    'xgb__reg_lambda': [0.5, 1],
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [5, 10]
}

# Using StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
grid_search = GridSearchCV(voting_classifier, param_grid, cv=skf, scoring='recall', n_jobs=-1, verbose=1)
grid_search.fit(X_train_sm_rfe, y_train_sm)

y_probs = grid_search.best_estimator_.predict_proba(X_test_rfe)[:,1]

# Post-processing: Adjusting classification threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
threshold = thresholds[np.argmax(2*(precision*recall)/(precision+recall))]  # F1 optimized threshold
y_preds = [1 if prob > threshold else 0 for prob in y_probs]

print(classification_report(y_test, y_preds))

# AUC of the precision-recall curve
auc_score = auc(recall, precision)
print(f"AUC of Precision-Recall Curve: {auc_score:.2f}")

grid_search.best_estimator_

# Extract the trained estimators from the VotingClassifier
trained_xgb = grid_search.best_estimator_.named_estimators_['xgb']
trained_rf = grid_search.best_estimator_.named_estimators_['rf']

# Obtain feature importances from each model
xgb_importances = trained_xgb.feature_importances_
rf_importances = trained_rf.feature_importances_

# Extract feature names from the RFE
selected_features = [X.columns[i] for i in range(len(rfe.support_)) if rfe.support_[i]]