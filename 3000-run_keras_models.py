"""
this script trains cy young prediction models with keras
"""

from numpy.random import seed
seed(42)

import os
import numpy as np
import pandas as pd

# working directory
wd = "/Users/ming-senwang/Dropbox/Current_Research/Predict_CY/"
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

def create_model():
    model = Sequential()
    model.add(layers.Dense(n, activation = 'relu', input_shape = (data.shape[1]-2, )))
    for j in range(l):
        model.add(layers.Dense(n, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', 
              optimizer = optimizers.RMSprop(lr = 0.01), 
              metrics = ['accuracy', ]
              )
    return model

out = []
for n in [16, 64, 256]:
    for l in range(3):
        model = create_model()
        err = []
        for s in season_set:
        	s0 = [item for item in season_set if item != s]
        	X, y = data_generator(data, s0, True)
        	X_val, y_val = data_generator(data, s, True)
        	model.fit(X, y, 
        		epochs = 100, batch_size = 50)
        	y_hat = model.predict(X_val).T.tolist()[0]
        	err += [(a - b) ** 2 for a, b in zip(y_val, y_hat)]
        out.append([l, n, np.sqrt(np.mean(err))])

res_nn = pd.DataFrame(out, columns = ['n_layers', 'layers', 'rmse'])
# =============================
# write results
res_nn.sort_values('rmse', inplace = True)
res_nn.to_csv(os.path.join(res_dir, 'nn_results.csv'), index = False)
