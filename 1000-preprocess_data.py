"""
preprocess data for cy young prediction
"""

import os
import numpy as np
import pandas as pd
import pybaseball as pb

# working directory
wd = "/Users/ming-senwang/Dropbox/Current_Research/Predict_CY/"
data_dir = "./data/"
os.chdir(wd)
qual = 50
last_season = 2017

# feature engineering
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
    dout = mod_data.drop(['Team', 'Dollars', 'Age Rng'], axis = 1).set_index('Season')
    # set Season as index so I can extract season for splitting data later
    print("\nwrite data to files\n")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dout.to_csv(os.path.join(data_dir, 'train_data.csv'), index = True)
