'''
Note: Code has been heavily adapted from Christopher V. Cosgriff, MD, MPH work on
sequential severity prediction for critically ill patients
(Source: https://github.com/cosgriffc/seq-severityscore)
'''

import numpy as np
import pandas as pd

# Pipeline
from sklearn.pipeline import Pipeline
# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
# Enable iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
# Models
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Save
import pickle

from sklearn.utils import resample
def upsample(X,y, ratio = 1.0):
	data = train_X
	data['hospital_expiration'] = train_y
	df_expired = data[data['hospital_expiration']==1]
	df_alive = data[data['hospital_expiration']==0]

	# upsample minority
	expired_upsampled = resample(df_expired,
	                          replace=True, # sample with replacement
	                          n_samples=int(ratio*len(df_alive)), # Adjust according to ratio
	                          random_state=42) # reproducible results

	# combine majority and upsampled minority
	upsampled = pd.concat([df_alive, expired_upsampled])

	# Return X and y
	y = upsampled['hospital_expiration'].values.ravel()
	X = upsampled.drop(['hospital_expiration'], axis=1)

	return X,y

def train_logit(X, y):

	logit_classifier = Pipeline([
		('impute', SimpleImputer(strategy='mean')),
		('scale', StandardScaler()),
		('logit', LogisticRegression(C=10**2, solver='lbfgs', max_iter=1000))])
	logit_classifier.fit(X, y)
	return logit_classifier

def train_svc(X, y):

	svc_classifier = Pipeline([
		('impute', SimpleImputer(strategy='mean')),
		('scale', StandardScaler()),
		('svc', SVC(C=10**2))])
	svc_classifier.fit(X, y)
	return svc_classifier

def train_rf_gridCV(X, y, params=None, K=10, n_iter=2000, upsample_ratio = 0, n_cpu=1, GPU=False):
	if params == None:
		params = {
		'max_depth': [3, 6, 9, 12],
	    'max_features': [0.1, 0.2, 0.4, 0.5],
	    'min_samples_leaf': [3, 4, 5],
	    'min_samples_split': [8, 10, 12],
	    'n_estimators': [100, 200, 300, 1000]
	}

	if upsample_ratio != 0:
		X, y = upsample(X,y, upsample_ratio)
		params['base_score'] = [sum(y) / len(y)]

	rf_classifier = Pipeline([
		('impute', SimpleImputer(strategy='mean')),
		('scale', StandardScaler()),
		('rf', RandomForestClassifier())])

	skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
	cv_grid_search = GridSearchCV(rf_classifier, param_grid=params,
		scoring='neg_log_loss', n_jobs=n_cpu, cv=skf.split(X, y),
		verbose=2)

	cv_grid_search.fit(X, y)
	rf_classifier = cv_grid_search.best_estimator_
	rf_classifier.fit(X, y)

	return rf_classifier

def train_xgb_gridCV(X, y, params=None, K=10, n_iter=2000, upsample_ratio = 0, n_cpu=1, GPU=False):
	if params == None:
		params = {'objective': ['binary:logistic'],
		'learning_rate': [0.005, 0.01, 0.05, 0.10],
		'max_depth': [3, 6, 9, 12],
		'min_child_weight': [6, 8, 10, 12],
		'silent': [True],
		'subsample': [0.6, 0.8, 1],
		'colsample_bytree': [0.5, 0.75, 1],
		'n_estimators': [500, 1000]}

	if GPU:
		xgb_classifier = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
	else:
		xgb_classifier = XGBClassifier()

	if upsample_ratio != 0:
		X, y = upsample(X,y, upsample_ratio)
		params['base_score'] = [sum(y) / len(y)]

	skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
	cv_grid_search = GridSearchCV(xgb_classifier, param_grid=params,
		scoring='roc_auc', n_jobs=n_cpu, cv=skf.split(X, y),
		verbose=2)

	cv_grid_search.fit(X, y)
	xgb_classifier = cv_grid_search.best_estimator_
	xgb_classifier.fit(X, y)

	return xgb_classifier

def save_model(model, file_name):
	pickle.dump(model, open('./{0}'.format(file_name), 'wb'))

# Load training data
train_X = pd.read_csv('../extraction/data/train_X.csv').set_index('patientunitstayid')
train_y = pd.read_csv('../extraction/data/train_y.csv').values.ravel()

#Train models
print('Training LogReg model.')
logit_full = train_logit(X=train_X, y=train_y)
save_model(logit_full, 'logit_full')
print('Done, model saved.')

print('Training SVC model.')
svc_full = train_svc(X=train_X, y=train_y)
save_model(svc_full, 'svc_full')
print('Done, model saved.')

rf_params = {
'rf__max_depth': [3, 4, 6, 8],
'rf__n_estimators': [50, 75, 100, 250, 500]
}
print('Training RF model.')
rf_full = train_rf_gridCV(X=train_X, y=train_y, params = rf_params, K=5, n_cpu=4, GPU=False)
save_model(rf_full, 'rf_full')
print('Done, model saved.')

# Search Parameter grid
#xgb_params = {
# 'objective': ['binary:logistic'],
# 'learning_rate': [0.005,0.01,0.02, 0.05,0.1],
# 'max_depth': [3,5,7,9],
# 'min_child_weight': [1,3,5],
# 'silent': [True],
# 'subsample': [0.5, 0.55, 0.6, 0.65, 0.7,0.75, 0.8],
# 'colsample_bytree': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
# 'n_estimators': [50, 100, 250, 500, 1000],
# 'gamma':[0, 0.1,0.2,0.3,0.4,0.5],
# }

# Final XGB Model
xgb_params = {
'objective': ['binary:logistic'],
'learning_rate': [0.005],
'max_depth': [5],
'min_child_weight': [1],
'silent': [True],
'subsample': [0.65],
'colsample_bytree': [0.75],
'n_estimators': [1000],
'gamma':[0.4]
}

print('Training XGB model.')
xgb_full = train_xgb_gridCV(X=train_X, y=train_y, params = xgb_params, K=5, n_cpu=4, GPU=False)
save_model(xgb_full, 'xgb_full')
print('Done, model saved.')
