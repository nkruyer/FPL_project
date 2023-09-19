# use the data pulled and added to in the FPL_data_pull to then do some ML analysis on
import os
import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import neighbors
from sklearn import neural_network
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression

GW = '4'
data_dir = '/Users/nick/PycharmProjects/FPL_project/data'
fname = f'performance_data_GW{GW}.csv'
infile = os.path.join(data_dir, fname)
dataset = pd.read_csv(infile)

stats_of_interest = ['element',
                     'total_points',
                     'opponent_team',
                     'was_home',
                     'opponent_strength',
                     'oppoenet_attack',
                     'opponent_defense',
                     'form',
                     'preround_total_points',
                     'transfers_reduced',
                     'average_points',
                     'preround_influence',
                     'preround_creativity',
                     'preround_threat',
                     'preround_ict',
                     'value'
                     ]

ML_in = dataset[stats_of_interest].sort_values(by = ['element'])

home = []
for i in list(ML_in['was_home']):
    if i:
        home.append(1)
    else:
        home.append(0)
ML_in['was_home'] = home

# first test Support Vector Regression
# turn into a pipeline?
# doing this agnostic to the player (element)

# build input and output arrays
pts = np.array(ML_in['total_points'])
param_df = ML_in.drop(labels = ['element','total_points'], axis = 1)
params = param_df.to_numpy()

# break into train and test sets, use default parameters for breakdown
X_train, X_test, y_train, y_test = train_test_split(params, pts)

# train a support vector regression model
regr = svm.SVR()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# get some other metrics?

r2_score(y_test, y_pred) # came out BBBAAADDDDD

# test with removing every player's first appearance
remove_firsts_df = ML_in[ML_in['form'] != 0].reset_index(drop = True).drop(labels = ['element'], axis = 1)

def standard_scaler(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled

def range_scaler(X_train, X_test):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled

# write function to do the dataframe --> test and train sets
def prepare_model_input(input_df):
    stats_of_interest = ['element',
                         'total_points',
                         'opponent_team',
                         'was_home',
                         'opponent_strength',
                         'opponent_attack',
                         'opponent_defense',
                         'form',
                         'preround_total_points',
                         'transfers_reduced',
                         'average_points',
                         'preround_influence',
                         'preround_creativity',
                         'preround_threat',
                         'preround_ict',
                         'value'
                         ]

    ML_in = input_df[stats_of_interest].sort_values(by=['element'])

    home = []
    for i in list(ML_in['was_home']):
        if i:
            home.append(1)
        else:
            home.append(0)
    ML_in['was_home'] = home

    # remove element column
    ML_in = ML_in.drop(labels=['element'], axis=1)

    # remove first appearances
    ML_in = ML_in[ML_in['form'] != 0]

    return ML_in

# K Neighbors Regressor
def build_knn(input_df, predicted_value = 'total_points', scaler_type = 'standard', weights = 'uniform', n_neighbors = 5, feature_selection = r_regression):
    #X = input_df.drop(labels=[predicted_value], axis=1).to_numpy()
    #y = np.array(input_df[predicted_value])
    #
    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    #
    #if scaler_type == 'standard':
    #    scaler, X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)
    #elif scaler_type == 'range':
    #    scaler, X_train_scaled, X_test_scaled = range_scaler(X_train, X_test)
    #else:
    #    print('scaler_type input undefined, used STANDARD')
    #    scaler, X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)
    #
    regr = neighbors.KNeighborsRegressor(n_neighbors, weights = weights)

    data, scaler, feature_selector = prepare_test_train(input_df, predicted_value=predicted_value,
                                                        scaler_type=scaler_type, feature_selection=feature_selection)

    if (scaler_type is not None) & (feature_selection is not None):
        d_type = 'reduced'
    elif (scaler_type is not None) & (feature_selection is None):
        d_type = 'scaled'
    else:
        d_type = 'raw'

    X_train = data['X_train'][d_type]
    y_train = data['y_train']['raw']
    regr.fit(X_train, y_train)

    return regr, scaler, feature_selector, d_type, data

# support vector regression
def build_svr(input_df, predicted_value = 'total_points', scaler_type = 'standard', kernel_type = 'rbf', feature_selection = r_regression): # can eventually add in parameters for the test/train split
    #X = input_df.drop(labels = [predicted_value], axis = 1).to_numpy()
    #y = np.array(input_df[predicted_value])

    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    #
    #if scaler_type == 'standard':
    #    scaler, X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)
    #elif scaler_type == 'range':
    #    scaler, X_train_scaled, X_test_scaled = range_scaler(X_train, X_test)
    #else:
    #    print('scaler_type input undefined, used STANDARD')
    #    scaler, X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)

    regr = svm.SVR(kernel = kernel_type)

    data, scaler, feature_selector = prepare_test_train(input_df, predicted_value=predicted_value,
                                                        scaler_type=scaler_type, feature_selection=feature_selection)

    if (scaler_type is not None) & (feature_selection is not None):
        d_type = 'reduced'
    elif (scaler_type is not None) & (feature_selection is None):
        d_type = 'scaled'
    else:
        d_type = 'raw'

    X_train = data['X_train'][d_type]
    y_train = data['y_train']['raw']
    regr.fit(X_train, y_train)

    return regr, scaler, feature_selector, d_type, data

# decision tree regression
def build_dtr(input_df, predicted_value = 'total_points', scaler_type = 'standard', criterion = 'squared_error', feature_selection = r_regression):
    #X = input_df.drop(labels=[predicted_value], axis=1).to_numpy()
    #y = np.array(input_df[predicted_value])

    #X_train, X_test, y_train, y_test = train_test_split(X, y)

    #if scaler_type == 'standard':
    #    scaler, X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)
    #elif scaler_type == 'range':
    #    scaler, X_train_scaled, X_test_scaled = range_scaler(X_train, X_test)
    #else:
    #    print('scaler_type input undefined, used STANDARD')
    #    scaler, X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)

    regr = tree.DecisionTreeRegressor(criterion=criterion)

    data, scaler, feature_selector = prepare_test_train(input_df, predicted_value = predicted_value,
                                                        scaler_type = scaler_type, feature_selection = feature_selection)

    if (scaler_type is not None) & (feature_selection is not None):
        d_type = 'reduced'
    elif (scaler_type is not None) & (feature_selection is None):
        d_type = 'scaled'
    else:
        d_type = 'raw'

    X_train = data['X_train'][d_type]
    y_train = data['y_train']['raw']
    regr.fit(X_train, y_train)

    return regr, scaler, feature_selector, d_type, data

# Multi-layer perceptron regressor - just using all default settings for now
def build_mlp(input_df, predicted_value = 'total_points', scaler_type = 'standard', max_iter = 500, feature_selection = r_regression):
    regr = neural_network.MLPRegressor(max_iter = max_iter)

    data, scaler, feature_selector = prepare_test_train(input_df, predicted_value=predicted_value,
                                                        scaler_type=scaler_type, feature_selection=feature_selection)

    if (scaler_type is not None) & (feature_selection is not None):
        d_type = 'reduced'
    elif (scaler_type is not None) & (feature_selection is None):
        d_type = 'scaled'
    else:
        d_type = 'raw'

    X_train = data['X_train'][d_type]
    y_train = data['y_train']['raw']
    regr.fit(X_train, y_train)

    return regr, scaler, feature_selector, d_type, data

def prepare_test_train(input_df, predicted_value = 'total_points', scaler_type = 'standard', feature_selection = None):
    data_dict = {}
    X = input_df.drop(labels = [predicted_value], axis = 1).to_numpy()
    y = np.array(input_df[predicted_value])

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    if scaler_type == 'standard':
        scaler, X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)
    elif scaler_type == 'range':
        scaler, X_train_scaled, X_test_scaled = range_scaler(X_train, X_test)
    elif scaler_type is None:
        scaler = None
        X_train_scaled = None
        X_test_scaled = None
    else:
        print('scaler_type input undefined, used STANDARD')
        scaler, X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)

    if feature_selection is None:
        feature_selector = None
        X_train_reduced = None
        X_test_reduced = None
    else:
        feature_selector, X_train_reduced = feature_reduction(X_train_scaled, y_train, k = 6, scoring_func = feature_selection)
        X_test_reduced = feature_selector.transform(X_test_scaled)

    data_dict['X_train'] = {'raw': X_train,
                            'scaled': X_train_scaled,
                            'reduced': X_train_reduced}
    data_dict['X_test'] = {'raw': X_test,
                           'scaled': X_test_scaled,
                           'reduced': X_test_reduced}
    data_dict['y_train'] = {'raw': y_train}
    data_dict['y_test'] = {'raw': y_test}

    return data_dict, scaler, feature_selector

# implement sklearn feature selection
def feature_reduction(X, y, k = 6, scoring_func = r_regression):
    if k > X.shape[1]:
        k = int(X.shape[1]/2)
    else:
        pass

    selection_obj = SelectKBest(scoring_func, k = k)
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)
    selection_obj.fit(X,y)
    X_new = selection_obj.transform(X)

    return selection_obj, X_new

# not sure yet if this will make the code easier/cleaner
class ML_data:
    def __init__(self, input_df):
        pass

