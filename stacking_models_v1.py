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
from sklearn.decomposition import PCA


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
    x_test = test_data.ix[:, 'ID':'X385']
    id_test = test_data.ix[:, 'ID']

    y_train = train_data['y']
    del train_data['y']
    x_train = train_data.ix[:, 'ID':'X385']

    print('\n\n Head')
    print("Test data")
    print(x_test.head())
    print("Train data")
    print(x_train.head())

    # id_train = train_data.ix[:,'ID']

    n_train, n_feat = x_train.shape  # 4209, 376
    n_test = x_test.shape[0]  # 4209

    # CATEGORICAL FEATURES MANIPULATION
    # check for values that are provided in test set but not in training set
    n_encode = 8  # the first 8 features are encoded
    columns = list(x_train.columns.values)
    categ_colum = columns[1:n_encode + 1]
    print(categ_colum)

    print_categorical_values(x_train, x_test, categ_colum)

    # ENCODING Categorical Features
    # CONVERT CATEGORICAL DATA INTO INTEGERS.
    # Hypothesis: letters encode sequencial information: a,..,z,aa,...az
    x_train, x_test = categ_2_int(x_train, x_test, categ_colum)

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

# ////////////////////////////////////////SOLVING////////////////////////////////////////
    # LOAD TRAINED MODELS
    nameModel = 'stack_NN_GBR_LR'
    model_GBR = joblib.load('models/grad_boosting_reg.pkl')
    model_LR, pca_lr = joblib.load('models/linear_reg.pkl')
    model_NN, min_max_scaler, pca_nn = joblib.load('models/neural_net_opt_reg.pkl')

    # MAKE INDEPENDENT DECISIONS
    # Gradient Boosting
    y_pred_GBR = test_model(x_test, model_GBR)

    # Neural Networks
    x_test_nn = min_max_scaler.transform(x_test)
    x_test_nn = pca_nn.transform(x_test_nn)
    y_pred_nn = test_model(x_test_nn, model_NN)

    # Linear Regression
    x_test_lr = pca_lr.transform(x_test)
    y_pred_LR = test_model(x_test_lr, model_LR)

    y_pred = 0.5*y_pred_GBR + 0.3*y_pred_LR + 0.2*y_pred_nn

# //////////////////////////////////////////////////////////////////////////////////////////////
    # GENERATE TEST REPORT
    # create dataframes
    id_test_df = pd.DataFrame(data=id_test.astype(np.int32), columns=['ID'], dtype=(int))
    y_pred_test_df = pd.DataFrame(data=y_pred, columns=['y'], dtype=(float))
    print(id_test_df.head())
    print(y_pred_test_df.head())

    # combine ID and y
    output_df = pd.concat([id_test_df, y_pred_test_df], axis=1)

    # create output file
    output_df.to_csv(path_or_buf='output/prediction_' + nameModel + '.csv', index=False, sep=',')



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