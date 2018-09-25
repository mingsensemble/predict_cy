#!/home/ubuntu/anaconda/bin/python3
"""
this script is designed to run on aws
it trains models with different machine learning algorithms
"""
from numpy.random import seed
seed(42)

import os
import numpy as np
import pandas as pd

# working directory
wd = "/home/ubuntu/predict_cy/"
data_dir = "./data/"
res_dir = "./results/"
os.chdir(wd)
# ====================================
data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'), index_col = 0)

def standardize(data):
    mu = np.mean(data)
    sigma = np.sqrt(np.var(data))
    out = [(x - mu)/sigma for x in data]
    return out

from sklearn.preprocessing import StandardScaler
def data_generator(data, season = None, scale = False):
    if season != None:
        data = data.loc[season]
    target = data['CY']
    sample = data.drop(['CY', 'Name'], axis = 1)
    if scale == True:
        scaler = StandardScaler()
        sample = scaler.fit_transform(sample)
    return sample, target
# ====================================
season_set = list(set(data.index))
# ====================================
# gbm
params = {
    'learning_rate': [0.1, 0.05, 0.01, 0.001],
    'n_estimators':[10, 100, 500, 1000],
    'max_depth': [1, 2, 3, 5, None]
}
from sklearn.ensemble import GradientBoostingClassifier
out = []
for l in params['learning_rate']:
	for n in params['n_estimators']:
		for d in params['max_depth']:
			gbm = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              loss='exponential', max_features = 'sqrt', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0,
              presort='auto', random_state=42, subsample=1.0, verbose=1,
              warm_start=False, 
              learning_rate = l, n_estimators = n, max_depth = d,
              )
			err = []
			# cross validation by season
			for s in season_set:
				X_val, y_val = data_generator(data, s, True)
				s0 = [item for item in season_set if item != s]
				X, y = data_generator(data, s0, True)
				gbm.fit(X, y)
				y_hat = gbm.predict_proba(X_val)[:, 1]
				err += [(a - b)  ** 2 for a, b in zip(y_val, y_hat)]
			out.append([l, n, d, np.sqrt(np.mean(err))])
res_gbm = pd.DataFrame(out, columns = ['learning_rate', 'n_estimators', 'max_depth', 'rmse'])
# =================================
# random forest
params = {
    'max_features': ('sqrt', 'log2'),
    'n_estimators':[10, 100, 500, 1000],
    'max_depth': [1, 2, 3, 5, None]
}
from sklearn.ensemble import RandomForestClassifier
out = []
for l in params['max_features']:
	for n in params['n_estimators']:
		for d in params['max_depth']:
			rfc = RandomForestClassifier(
              max_features = l, n_estimators = n, max_depth = d,
              )
			err = []
			# cross validation by season
			for s in season_set:
				X_val, y_val = data_generator(data, s, True)
				s0 = [item for item in season_set if item != s]
				X, y = data_generator(data, s0, True)
				rfc.fit(X, y)
				y_hat = rfc.predict_proba(X_val)[:, 1]
				err += [(a - b)  ** 2 for a, b in zip(y_val, y_hat)]
			out.append([l, n, d, np.sqrt(np.mean(err))])
res_rfc = pd.DataFrame(out, columns = ['max_features', 'n_estimators', 'max_depth', 'rmse'])
# ======================================
# support vector machine
params = {
    'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
    'C':[0.1, 0.5, 1, 5, 100],
}
from sklearn.svm import SVC
out = []
for k in params['kernel']:
	for c in params['C']:
		svm = SVC(
          kernel = k, C = c, max_iter = 1e3, random_state = 42, probability = True
          )
		err = []
		# cross validation by season
		for s in season_set:
			X_val, y_val = data_generator(data, s, True)
			s0 = [item for item in season_set if item != s]
			X, y = data_generator(data, s0, True)
			svm.fit(X, y)
			y_hat = svm.predict_proba(X_val)[:, 1]
			err += [(a - b)  ** 2 for a, b in zip(y_val, y_hat)]
		out.append([k, c, np.sqrt(np.mean(err))])
res_svm = pd.DataFrame(out, columns = ['kernel', 'C', 'rmse'])
# ======================================
# collect results
res_gbm.sort_values('rmse', inplace = True)
res_gbm.to_csv(os.path.join(res_dir, 'gbm_results.csv'), index = False)
res_rfc.sort_values('rmse', inplace = True)
res_rfc.to_csv(os.path.join(res_dir, 'rfc_results.csv'), index = False)
res_svm.sort_values('rmse', inplace = True)
res_svm.to_csv(os.path.join(res_dir, 'svm_results.csv'), index = False)
