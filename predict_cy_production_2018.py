"""
predicts Cy Youn award winner -- production code
"""
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

import os
import numpy as np
import pandas as pd
import pybaseball as pb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--working_directory', type = str, default = 'predict_cy/',
    help = 'define working directory'
    )
parser.add_argument(
    '--data_dir', type = str, default = 'Production',
    help = 'Directory to store modeling data'
)
parser.add_argument(
    '--ensemble', type = int, default = 0,
    help = 'Whether to ensemble multiple models, default to 0'
    )
parser.add_argument(
    '--top_k', type = int, default = 10,
    help = 'Report top k scorers, default to 10'
    )
parser.add_argument(
    '--download_data', type = int, default = 1,
    help = 'Whether to download data, default to 1 (yes)'
    )
parser.add_argument(
    '--projection_season', type = int, default = 2018,
    help = 'define projection season'
    )
parser.add_argument(
    '--qual', type = int, default = 50,
    help = 'minimum qualified innings pitched, default = 50'
    )

# manage data
def create_var(data):
    """ create variables for modeling"""
    data['Start-IP%'] = data['Start-IP']/data['IP']
    data['Start-IP%'].fillna(0, inplace = True)
    data['FB%'].fillna(0, inplace = True)
    data['netGB%'] = data['GB%'] - data['FB%']
    data['netGB%'].fillna(0, inplace = True)
    data['Strike%'] = data['Strikes']/data['Pitches']
    data['HR/9'] = 9 * data['HR']/data['IP']
    data['HR%'] = data['HR']/data['TBF']
    data['WP/9'] = 9 * data['WP']/data['IP']
    data['WP%'] = data['WP']/data['Pitches']
    data['GS%'] = data['GS']/data['G']
    data['GS%'].fillna(0, inplace = True)
    data['Hard-Soft%'] = data['Hard%'] - data['Soft%']
    data['K%2'] = data['K%'] ** 2
    data['BB%2'] = data['BB%'] ** 2
    data['netGB%2'] = data['netGB%'] ** 2
    data['K_BB%'] = data['K%'] * data['BB%']
    data['netGB_K%'] = (data['GB%'] - data['FB%']) * data['K%']
    data['netGB_BB%'] = (data['GB%'] - data['FB%']) * data['BB%']
    data['IP/GS'] = data['IP']/data['GS']
    data['IP/GS'].fillna(0, inplace = True)
    data['IP/GS'].replace(np.inf, 0, inplace = True)
    data['HR/IP'] = data['HR']/data['IP']
    data['BB/IP'] = (data['BB'] + data['HBP'])/data['IP']
    data['K/IP'] = data['SO']/data['IP']
    data['Starting'].fillna(0, inplace = True)
    data['Start-IP'].fillna(0, inplace = True)
    data['Relieving'].fillna(0, inplace = True)
    data['Relief-IP'].fillna(0, inplace = True)
    data = data.set_index('Season')
    data.fillna(0, inplace = True)
    return data

def preprocess_data():
    last_season = FLAGS.projection_season
    qual = FLAGS.qual
    # read in CY historical data
    print("\nload cy young award winner data")
    append_data = []
    # i have Cy Young winner data from 2006 to 2015
    for i in range(2006, 2016):
        for j in ['AL', 'NL']:
            file_name = "CY_" + str(j) +"_" + str(i) + ".txt"
            tmp = pd.read_table(os.path.join('Data', file_name), sep = ",")
            tmp['Season'] = i
            tmp['League'] = j
            append_data.append(tmp)
    old_cy = pd.concat(append_data, axis = 0)
    # winner of CY has Rank 1
    old_cy['CY'] = [int(x == 1) for x in old_cy['Rank']]
    cy = old_cy[['Name', 'Season', 'CY']]
    # download baseball data: start from 2006 because more advanced stats are recorded since
    print("\nstart downloading data")
    stats1 = pb.pitching_stats(start_season = 2006, end_season = 2010, qual = qual)
    stats2 = pb.pitching_stats(start_season = 2011, end_season = last_season, qual = qual)
    df = create_var(pd.concat([stats1, stats2], axis = 0))
    data = pd.merge(df, cy, on = ['Name', 'Season'], how = 'outer')
    data['CY'].fillna(0, inplace = True)
    # enter 2016 and 2017 winners
    data.loc[[all([any([a, b]), c]) for a, b, c in zip(data.Name == 'Max Scherzer', data.Name == 'Rick Porcello', data.Season == 2016)], 'CY'] = 1
    data.loc[[all([any([a, b]), c]) for a, b, c in zip(data.Name == 'Max Scherzer', data.Name == 'Corey Kluber', data.Season == 2017)], 'CY'] = 1    
    # split training and testing data
    # drop columns if it has at least one missing value
    print("\npreprocess data")
    mod_data = data.dropna(thresh = data.shape[0], axis = 1)
    X = mod_data.drop(['CY', 'Name', 'Team', 'Dollars', 'Age Rng'], axis = 1).set_index('Season')
    # set Season as index so I can extract season for splitting data later
    y = mod_data[['CY', 'Name']].set_index('Name')
    X_train_raw, X_pred_raw, y_train, y_pred = X[X.index != last_season], X[X.index == last_season], y[X.index != last_season], y[X.index == last_season]
    # scale data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_pred = scaler.fit_transform(X_pred_raw)
    print("\nwrite data to files\n")
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    pd.DataFrame(X_train, columns = X.columns, index = X_train_raw.index).to_csv(os.path.join(FLAGS.data_dir, 'train.csv'), index = True)
    pd.DataFrame(X_pred, columns = X.columns, index = X_pred_raw.index).to_csv(os.path.join(FLAGS.data_dir, 'pred.csv'), index = True)
    pd.DataFrame(y_train).to_csv(os.path.join(FLAGS.data_dir, 'train_label.csv'), index = True) 
    pd.DataFrame(y_pred.index).to_csv(os.path.join(FLAGS.data_dir, 'pred_label.csv'), index = False) 

def train_and_predict():
    X = pd.read_csv(os.path.join(FLAGS.data_dir, 'train.csv'), index_col = 0)
    y = pd.read_csv(os.path.join(FLAGS.data_dir, 'train_label.csv')).iloc[:,1]
    X_pred = pd.read_csv(os.path.join(FLAGS.data_dir, 'pred.csv'), index_col = 0).values.astype('float32')
    name_pred = pd.read_csv(os.path.join(FLAGS.data_dir, 'pred_label.csv'))
    from keras.models import Sequential
    from keras import optimizers
    from keras import layers
    def create_model():
        model = Sequential()
        model.add(layers.Dense(16, activation = 'relu', input_shape = (X.shape[1], )))
        model.add(layers.Dense(16, activation = 'relu', input_shape = (X.shape[1], )))
        model.add(layers.Dense(1, activation = 'sigmoid'))
        model.compile(loss = 'binary_crossentropy', 
                  optimizer = optimizers.RMSprop(lr = 0.01), 
                  metrics = ['accuracy', ]
                  )
        return model
    out = []
    print("\nestimate neural network ...")
    for i in range(30):
        print("\niteration %d"%i)
        nn = create_model()
        nn.fit(X.values.astype('float32'), y.values.astype('float32'), epochs = 100, batch_size = 512, verbose = 0)
        out.append(nn.predict(X_pred))
    from functools import reduce 
    nn_pred = [z/30 for z in reduce(lambda x, y: [(a + b) for a, b in zip(x, y)], out)]
    if FLAGS.ensemble > 0:
        from sklearn.ensemble import GradientBoostingClassifier
        print("\nestimate gradient boosting machine ...")
        gbm = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              loss='exponential', max_features = 'sqrt', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0,
              presort='auto', random_state=42, subsample=1.0, verbose=0,
              warm_start=False, 
              learning_rate = 0.01, n_estimators = 1000, max_depth = 3,
            )
        gbm.fit(X.values.astype('float32'), y.values.astype('float32'))
        from sklearn.ensemble import RandomForestClassifier
        print("\nestimate random forest ...")
        rfc = RandomForestClassifier(
              max_features = 'sqrt', n_estimators = 500, max_depth = 5,
            )
        rfc.fit(X.values.astype('float32'), y.values.astype('float32'))
        from sklearn.svm import SVC
        print("\nestimate support vector machine ...")
        svm = SVC(
          kernel = 'rbf', C = 100, max_iter = 1e3, random_state = 42, probability = True
          )
        svm.fit(X.values.astype('float32'), y.values.astype('float32'))
        y_hat = pd.DataFrame([
            nn.predict(X).T.tolist()[0],
            gbm.predict_proba(X)[:, 1].tolist(),
            rfc.predict_proba(X)[:, 1].tolist(),
            svm.predict_proba(X)[:, 1].tolist(),
        ]).T
        # model stacking
        from sklearn.linear_model import LogisticRegression
        print("\nstack models ...")
        lr = LogisticRegression(C = 5.0, max_iter = 10000) 
        lr.fit(y_hat, y)

        res = pd.concat([
            name_pred, 
            pd.DataFrame(nn_pred),
            pd.DataFrame(gbm.predict_proba(X_pred)).iloc[:, 1],
            pd.DataFrame(rfc.predict_proba(X_pred)).iloc[:, 1],
            pd.DataFrame(svm.predict_proba(X_pred)).iloc[:, 1],            
        ], axis = 1)
        stack_pred = lr.predict_proba(res.iloc[:, -4:])[:, 1]
        res.columns = ['name', 'nn_score', 'gbm_score', 'rfc_score', 'svm_score']
        res['score'] = stack_pred
    else:
        res = pd.concat([
            name_pred, 
            pd.DataFrame(nn_pred),
        ], axis = 1)
        res.columns = ['name', 'score']
    res.sort_values(axis = 0, by = 'score', ascending = False).head(FLAGS.top_k).to_csv(os.path.join(FLAGS.data_dir, 'projection_results.csv'), index = False) 

def main():
    # working directory
    os.chdir(FLAGS.working_directory)
    if FLAGS.download_data == 1:
        print('\ndownloading data ... ...')
        preprocess_data()
        print('\ndownload complete!')
    print('\nestimating model ... ...')
    train_and_predict()
    print('\nestimation complete!')
    print('\noutput results')
    
if __name__ == '__main__':
    FLAGS = parser.parse_args()
    main()
