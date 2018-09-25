"""
uses logistic regression to stack the best performing models 
in each algorithm
"""
from numpy.random import seed
seed(42)

import os
import numpy as np
import pandas as pd

# working directory
wd = "predict_cy/"
data_dir = "./data/"
os.chdir(wd)

data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'), index_col = 0)

from sklearn.preprocessing import StandardScaler
def data_generator(data, season = None, scale = False):
    if season != None:
        data = data.loc[season]
    target = data['CY']
    sample = data.drop(['CY', 'Name'], axis = 1)
    if scale == True:
        scaler = StandardScaler()
        sample = scaler.fit_transform(sample)
    return sample.astype('float32'), target
# ====================================
season_set = list(set(data.index))
# ====================================
from keras.models import Sequential
from keras import optimizers
from keras import layers

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def stack_models(X, y, X_pred):
    # train models with X, y
    # neural network
    def create_model():
        model = Sequential()
        model.add(layers.Dense(16, activation = 'relu', input_shape = (data.shape[1]-2, )))
        model.add(layers.Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'binary_crossentropy', 
                  optimizer = optimizers.RMSprop(lr = 0.01), 
                  metrics = ['accuracy', ]
                  )
        return model
    nn = create_model()
    # GBM
    gbm = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              loss='exponential', max_features = 'sqrt', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0,
              presort='auto', random_state=42, subsample=1.0, verbose=0,
              warm_start=False, 
              learning_rate = 0.01, n_estimators = 1000, max_depth = 3,
            )
    # random forest
    rfc = RandomForestClassifier(
              max_features = 'sqrt', n_estimators = 500, max_depth = 5,
            )
    # support vector machine
    svm = SVC(
          kernel = 'rbf', C = 100, max_iter = 1e3, random_state = 42, probability = True
          )
    print("fit neural network ...")
    nn.fit(X, y, epochs = 100, batch_size = 50, verbose = 0)
    print("fit GBM ...")
    gbm.fit(X, y)
    print("fit random forest ...")
    rfc.fit(X, y)
    print("fit suport vector machine ...")
    svm.fit(X, y)
    y_hat = pd.DataFrame([
        nn.predict(X).T.tolist()[0],
        gbm.predict_proba(X)[:, 1].tolist(),
        rfc.predict_proba(X)[:, 1].tolist(),
        svm.predict_proba(X)[:, 1].tolist(),
      ]).T
    print("stack models ...")
    lr = LogisticRegression(C = c, max_iter = 10000)
    lr.fit(y_hat, y)
    # prediction
    y_pred = pd.DataFrame([
        nn.predict(X_pred).T.tolist()[0],
        gbm.predict_proba(X_pred)[:, 1].tolist(),
        rfc.predict_proba(X_pred)[:, 1].tolist(),
        svm.predict_proba(X_pred)[:, 1].tolist(),
      ]).T
    res = lr.predict_proba(y_pred)[:, 1].tolist()
    return res

# ==============================
# model stacking
season_set = list(set(data.index))
out = []
for c in [0.1, 0.5, 1, 5, 10, 100]:
    err = []
    for s in season_set:
        s0 = [item for item in season_set if item != s]
        X, y = data_generator(data, s0, True)
        X_val, y_val = data_generator(data, s, True)
        y_hat = stack_models(X, y, X_val)
        err += [(a - b) ** 2 for a, b in zip(y_val, y_hat)]
    out.append([c, np.sqrt(np.mean(err))])

res_stack = pd.DataFrame(out, columns = ['c', 'rmse'])
res_stack.sort_values('rmse', inplace = True)
res_stack.to_csv(os.path.join(res_dir, 'stack_results.csv'), index = False)