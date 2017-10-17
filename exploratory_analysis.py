import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import Regressors as reg
import Visualizations as visio
from sklearn.externals import joblib
from sklearn import preprocessing
import xgboost as xgb

def main():
    # INITIAL DATA MANIPULATION
    # load data
    test_data = pd.read_csv('data/test.csv')
    train_data = pd.read_csv('data/train.csv')
    # data structure and general info

    print("Test data")
    print(test_data.head())
    print("Train data")
    print(train_data.head())

    print("Test data")
    print(test_data.info())
    print("Train data")
    print(train_data.info())

    # consider that ID column also affects the output (slightly decreasing output as increasing ID)
    x_test = test_data.ix[:,'ID':'X385']
    id_test = test_data.ix[:,'ID']

    y_train = train_data['y']
    del train_data['y']
    x_train = train_data.ix[:,'ID':'X385']

    print('\n\n Head')
    print("Test data")
    print(x_test.head())
    print("Train data")
    print(x_train.head())

    # id_train = train_data.ix[:,'ID']

    n_train,n_feat = x_train.shape      # 4209, 376
    n_test = x_test.shape[0]            # 4209

    # CATEGORICAL FEATURES MANIPULATION
    # check for values that are provided in test set but not in training set
    n_encode = 8    # the first 8 features are encoded
    columns = list(x_train.columns.values)
    categ_colum = columns[1:n_encode+1]
    print(categ_colum)

    print_categorical_values(x_train, x_test, categ_colum)

    # ENCODING Categorical Features
    # CONVERT CATEGORICAL DATA INTO INTEGERS.
    # Hypothesis: letters encode sequencial information: a,..,z,aa,...az
    x_train,x_test = categ_2_int(x_train,x_test,categ_colum)

    # FEATURE SCALING


    # DATA VISUALIZATIONS
    # visualize y
    # visualize_y_ID(x_train,y_train)
    # visualize_y_categorical(x_train.ix[:,'X0':'X8'],y_train)
    # histogram_y(y_train)

    # convert to Numpy
    x_train = x_train.as_matrix()
    y_train = y_train.as_matrix()
    x_test = x_test.as_matrix()
    id_test = id_test.as_matrix()


    # TRAIN
    k_fold = 10
    approach = 7

    if approach == 0:
        # SVM REGRESSOR
        model = reg.train_svm_regressor(x_train, y_train,k_fold)
        nameModel = 'svm_reg'

    elif approach == 1:
        # LARS REGRESSION
        model = reg.train_lars_regressor(x_train, y_train,k_fold)
        nameModel = 'lars_reg'
        joblib.dump((model), 'models/' + nameModel + '.pkl')

    elif approach == 2:
        # LASSO REGRESSION
        model = reg.train_lasso_regressor(x_train, y_train, k_fold)
        nameModel = 'lasso_reg'
        joblib.dump((model), 'models/' + nameModel + '.pkl')

    elif approach == 3:
        model = reg.train_grad_boost_regressor(x_train, y_train, k_fold)
        nameModel = 'grad_boosting_reg'
        joblib.dump((model), 'models/' + nameModel + '.pkl')

    elif approach == 4:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train_scaled = min_max_scaler.fit_transform(x_train)
        x_test = min_max_scaler.transform(x_test)

        model, pca = reg.tune_linear_model(x_train_scaled, y_train, k_fold)
        nameModel = 'linear_reg'
        if pca !=-1:
            x_test = pca.transform(x_test)
        joblib.dump((model, min_max_scaler, pca), 'models/' + nameModel + '.pkl')

    elif approach == 5:
        n_iter = 500
        model = reg.tune_grad_boost_regressor(x_train, y_train, k_fold, n_iter)
        nameModel = 'grad_boosting_opt_reg'
        # Save model
        joblib.dump((model ), 'models/' + nameModel + '.pkl')

    elif approach == 6:
        n_iter = 500
        # re-scaling
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train_scaled = min_max_scaler.fit_transform(x_train)
        x_test = min_max_scaler.transform(x_test)

        model, pca = reg.tune_feed_forward_nn_regressor(x_train_scaled, y_train, k_fold, n_iter)
        nameModel = 'neural_net_opt_reg'
        if pca !=-1:
            x_test = pca.transform(x_test)

        # Save model
        joblib.dump((model, min_max_scaler, pca), 'models/' + nameModel + '.pkl')

    elif approach == 7:
        n_iter = 1000
        # re-scaling
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train_scaled = min_max_scaler.fit_transform(x_train)
        x_test = min_max_scaler.transform(x_test)

        model, pca = reg.tune_xgboost_regressor(x_train_scaled, y_train, k_fold, n_iter)
        nameModel = 'xgboost_opt_reg'
        if pca !=-1:
            x_test = pca.transform(x_test)
        x_test = xgb.DMatrix(x_test)

        # Save model
        joblib.dump((model, min_max_scaler, pca), 'models/' + nameModel + '.pkl')

    print('\n\nFeature Importance')
#    print(model.feature_importances_)
    # visio.visualize_feature_importance(model)


    # load with: model = joblib.load('filename.pkl')
# //////////////////////////////////////////////////////////////////////////////////////////////
    # TEST REGRESSOR
    y_pred_test = test_model(x_test, model)

    # GENERATE TEST REPORT
    # create dataframes
    id_test_df = pd.DataFrame(data=id_test.astype(np.int32), columns=['ID'],dtype=(int))
    y_pred_test_df = pd.DataFrame(data=y_pred_test,columns=['y'],dtype=(float))
    print(id_test_df.head())
    print(y_pred_test_df.head())

    # combine ID and y
    output_df = pd.concat([id_test_df, y_pred_test_df], axis=1)

    # create output file
    print(nameModel)
    output_df.to_csv(path_or_buf='output/prediction_' + nameModel + '.csv', index=False, sep=',')

    '''
    y_pred_final = np.column_stack([id_test,y_pred_test])
    print(y_pred_final.shape)
    # create dataframe
    y_pred_test = pd.DataFrame(data=y_pred_final, index=None, columns=['ID','y'],dtype=(int,float))
    print(y_pred_test.head())

    y_pred_test.to_csv(path_or_buf='output/prediction.csv', index=False, sep=',')
    '''






def test_model(x_test, model):
    y_pred = model.predict(x_test)
    return y_pred



def print_categorical_values(x_train, x_test, categ_colum):
    # values that are taken in the test set but not in the training set.
    for c in categ_colum:
        print(c, ':', end=' ')
        for e in x_test[c].unique():
            if e not in x_train[c].unique(): print(e, end=' ')
        print(' ')
    print(' ')

    # values that are taken in the training set but not in the test set.
    for c in categ_colum:
        print(c, ':', end=' ')
        for e in x_train[c].unique():
            if e not in x_test[c].unique(): print(e, end=' ')
        print(' ')


    print(' ####################################')
    print('Values for each one of the categorical values that we have')
    # What
    for c in categ_colum:
        print(c, ':', end=' ')
        for e in x_train[c].unique():
            print(e, end=' ')
        for e in x_test[c].unique():
            if e not in x_train[c].unique(): print(e, end=' ')
        print()


def categ_2_int(x_train, x_test, categ_colum):
    encoder = LabelEncoder()

    # fit classes based on the hypothetical order
    encoder.fit(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'aa', 'ab', 'ac', 'ad', 'ae', 'af',
                 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au',
                 'av', 'aw', 'ax', 'ay', 'az', 'ba', 'bb', 'bc'])

    # transform columns
    for c in categ_colum:
        # transform column
        colum_new_train = encoder.transform(x_train[c])
        colum_new_test = encoder.transform(x_test[c])

        # print(colum_new_train)
        # substitute dataframe column
        x_train[c] = colum_new_train
        x_test[c] = colum_new_test
        # print(x_train.head())

    return x_train, x_test

main()

