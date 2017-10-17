import numpy as np

from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from time import time
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.neural_network import MLPRegressor

import Visualizations as visio
import xgboost as xgb


def tune_xgboost_regressor(X, y, k_fold, n_iter_search):
    n_dim = 150
    x_train, pca_map = pca_d_reduction(X, n_dim)

    xgb_params = {
        'n_trees': 1000,
        'eta': 0.0045,
        'max_depth': 4,
        'subsample': 0.93,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        #'base_score': y_mean,  # base prediction = mean(target)
        'silent': 1
    }
    # NOTE: Make sure that the class is labeled 'class' in the data file

    dtrain = xgb.DMatrix(x_train, y)

    # train model
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=n_iter_search)

    return model, pca_map


def tune_feed_forward_nn_regressor(X, y, k_fold, n_iter_search):
    n_dim = 100
    x_train, pca_map = pca_d_reduction(X, n_dim)

    # feed-forward neural network regressor
    # Adaptive Momentum (ADAM)
    '''
    regressor = MLPRegressor(activation='relu', solver='adam',
                            alpha=0.0001, batch_size= 500,
                            learning_rate_init=0.001, max_iter=2000, shuffle=True,
                            random_state=42, tol=0.0001, verbose=True, warm_start=False,
                            early_stopping=False,
                             beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    '''
    # stochastic gradient descent
    regressor = MLPRegressor(activation='logistic', solver='sgd', alpha=0.01,
                             batch_size=1000, learning_rate='adaptive',
                             learning_rate_init=0.001, power_t=0.5, max_iter=3000,
                             shuffle=True, random_state=42, tol=0.0001, verbose=True,
                             warm_start=True, momentum=0.9, nesterovs_momentum=True,
                             early_stopping=False, validation_fraction=0.1)

    # specify parameters and distributions to sample from
    param_dist = {"hidden_layer_sizes": sp_randint(28,29)
                  }

    # run randomized search
    random_search = RandomizedSearchCV(estimator=regressor, param_distributions=param_dist,
                                       n_iter=n_iter_search, fit_params=None, n_jobs=1,
                                       iid=True, refit=True, cv=k_fold, verbose=1, pre_dispatch='2*n_jobs',
                                       random_state=None, error_score='raise', return_train_score=True)

    start = time()
    random_search.fit(x_train, y)
    print("\nRandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))

    # report results from
    report(random_search.cv_results_)

    model = random_search.best_estimator_

    print("The best score in the search process is %f" % random_search.best_score_)

    return model, pca_map

def tune_grad_boost_regressor(X, y, k_fold, n_iter_search):
    model_all = []
    r2_all = []
    r2_mean_all = []

    # Gradient Boosting Regressor
    #loss = 'ls'  # ls, huber, lad
    regressor = GradientBoostingRegressor(loss='ls')

    # specify parameters and distributions to sample from
    param_dist = {"learning_rate": sp_uniform(0,1),
                  "n_estimators": sp_randint(40,1500),
                  "max_depth": sp_randint(2,6),
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11)
                  }

    # run randomized search
    random_search = RandomizedSearchCV(estimator=regressor, param_distributions=param_dist,
                                       n_iter=n_iter_search, fit_params=None, n_jobs=1,
                                       iid=True, refit=True, cv=k_fold, verbose=1, pre_dispatch='2*n_jobs',
                                       random_state=None, error_score='raise', return_train_score=True)

    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))

    # report results from
    report(random_search.cv_results_)

    model = random_search.best_estimator_

    print("The best score in the search process is %f", random_search.best_score_)

    return model

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def tune_linear_model(X, y, k_fold):
    # REDUCE DIMENSIONALITY
    do_pca = True
    model_all = []
    pca_all = []
    r2_all = []
    r2_mean_all = []
    # n_dim = X.shape[1]
    n_dim = 255
    if do_pca == True:
        print('\nReduce Dimensionality')
        for dim in range(1, n_dim):
            print('\nFitting with %d', dim)
            x_train, pca_temp = pca_d_reduction(X, dim)

            model_temp, r2_temp, r2_mean_temp = train_linear_regressor(x_train, y, k_fold)

            model_all.append(model_temp)
            r2_all.append(r2_temp)
            r2_mean_all.append(r2_mean_temp)
            pca_all.append(pca_temp)

        # Validation R2 vs. nº of dimensions
        print(r2_mean_all)
        visio.visio_pca_dim(r2_mean_all, n_dim)

        idx = np.argmax(r2_mean_all)
        model = model_all[idx]
        pca_map = pca_all[idx]

        print('The best validation R2 is obtain with %d dimensions', idx)

    else:
        pca_map = -1
        model = train_linear_regressor(X, y, k_fold)

    return model, pca_map


def pca_d_reduction(x_train, n):
    pca = PCA(n_components= n, svd_solver='full')
    pca.fit(x_train)
    print('\nExplained Variance Ratio from PCA')
    # print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))

    x_train = pca.transform(x_train)

    return x_train, pca

def train_linear_regressor(X, y, k_fold):
    ##  CROSS VALIDATION
    # Without evenly distributing classes
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=None)
    kf.get_n_splits()

    print(kf)
    print(kf.get_n_splits())

    # Evenly Distributing Classes: Stratified K-fold
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # skf.get_n_splits(X, y)
    # print(skf)

    # Linear Regressor
    regressor = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=False)

    # k-Fold loop
    counter = 0
    model = []
    r2 = []
    print("\nShape")
    print(X.shape)
    print(y.shape)
    for train_index, test_index in kf.split(X, y=y):
        # print("TRAIN: ", train_index, " TEST: ", test_index)
        # get indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ### SVM CLASSIFIER ####################################
        # fit model
        model.append(regressor.fit(X_train, y_train))

        # predict
        y_pred = model[counter].predict(X_test)

        print("Coefficient of Determination for Regressor ", counter)
        r2.append(model[counter].score(X_test, y_test))
        print(r2[counter])

        counter += 1
    r2_mean = np.mean(r2)
    # choose svm with highest r2
    idx = np.argmax(r2)
    return model[idx], r2[idx], r2_mean

def train_grad_boost_regressor(X, y, k_fold):
    ##  CROSS VALIDATION
    # Without evenly distributing classes
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=None)
    kf.get_n_splits()

    print(kf)
    print(kf.get_n_splits())

    # Evenly Distributing Classes: Stratified K-fold
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # skf.get_n_splits(X, y)
    # print(skf)

    # Gradient Boosting Regressor
    loss = 'ls'  # ls, huber, lad
    regressor = GradientBoostingRegressor(loss=loss, n_estimators=1000, max_depth=3,
                                    learning_rate=.01)

    # k-Fold loop
    counter = 0
    model = []
    r2 = []
    print("\nShape")
    print(X.shape)
    print(y.shape)
    for train_index, test_index in kf.split(X, y=y):
        # print("TRAIN: ", train_index, " TEST: ", test_index)
        # get indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ### SVM CLASSIFIER ####################################
        # fit model
        model.append(regressor.fit(X_train, y_train))

        # predict
        y_pred = model[counter].predict(X_test)

        print("Coefficient of Determination for Regressor ", counter)
        r2.append(model[counter].score(X_test, y_test))
        print(r2[counter])

        counter += 1

    # choose svm with highest r2
    idx = np.argmax(r2)
    return model[idx]

def train_xgboost_regressor(X, y, k_fold):
    ##  CROSS VALIDATION
    # Without evenly distributing classes
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=None)
    kf.get_n_splits()

    print(kf)
    print(kf.get_n_splits())

    # Evenly Distributing Classes: Stratified K-fold
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # skf.get_n_splits(X, y)
    # print(skf)

    # XGBoost Regressor
    regressor = 0

    # k-Fold loop
    counter = 0
    model = []
    r2 = []
    print("\nShape")
    print(X.shape)
    print(y.shape)
    for train_index, test_index in kf.split(X, y=y):
        # print("TRAIN: ", train_index, " TEST: ", test_index)
        # get indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ### SVM CLASSIFIER ####################################
        # fit model
        model.append(regressor.fit(X_train, y_train))

        # predict
        y_pred = model[counter].predict(X_test)

        print("Coefficient of Determination for Regressor ", counter)
        r2.append(model[counter].score(X_test, y_test))
        print(r2[counter])

        counter += 1

    # choose svm with highest r2
    idx = np.argmax(r2)
    return model[idx]


def train_lasso_regressor(X, y, k_fold):
    ##  CROSS VALIDATION
    # Without evenly distributing classes
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=None)
    kf.get_n_splits()

    print(kf)
    print(kf.get_n_splits())

    # Evenly Distributing Classes: Stratified K-fold
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # skf.get_n_splits(X, y)
    # print(skf)

    # SVM classifier definition
    regressor = linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False,
                   precompute=False, copy_X=True, max_iter=1000, tol=0.0001,
                   warm_start=False, positive=False, random_state=None, selection='cyclic')

    # k-Fold loop
    counter = 0
    model = []
    r2 = []
    print("\nShape")
    print(X.shape)
    print(y.shape)
    for train_index, test_index in kf.split(X, y=y):
        # print("TRAIN: ", train_index, " TEST: ", test_index)
        # get indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ### SVM CLASSIFIER ####################################
        # fit model
        model.append(regressor.fit(X_train, y_train))

        # predict
        y_pred = model[counter].predict(X_test)

        print("Coefficient of Determination for Regressor ", counter)
        r2.append(model[counter].score(X_test, y_test))
        print(r2[counter])

        counter += 1

    # choose svm with highest r2
    idx = np.argmax(r2)
    return model[idx]


def train_svm_regressor(X, y, k_fold):
    ##  CROSS VALIDATION
    # Without evenly distributing classes
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=None)
    kf.get_n_splits()

    print(kf)
    print(kf.get_n_splits())

    # Evenly Distributing Classes: Stratified K-fold
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # skf.get_n_splits(X, y)
    # print(skf)

    # SVM classifier definition
    regressor = SVR(C=1.0, kernel='rbf', gamma='auto', verbose=False)


    # k-Fold loop
    counter = 0
    svm_model = []
    r2 = []
    print("\nShape")
    print(X.shape)
    print(y.shape)
    for train_index, test_index in kf.split(X, y=y):
        # print("TRAIN: ", train_index, " TEST: ", test_index)
        # get indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ### SVM CLASSIFIER ####################################
        # fit model
        svm_model.append(regressor.fit(X_train, y_train))

        # predict
        y_pred = svm_model[counter].predict(X_test)

        print("Coefficient of Determination for Regressor ", counter)
        r2.append(svm_model[counter].score(X_test, y_test))
        print(r2[counter])

        counter += 1

    # choose svm with highest r2
    idx = np.argmax(r2)
    return svm_model[idx]

def train_lars_regressor(X, y, k_fold):
    ##  CROSS VALIDATION
    # Without evenly distributing classes
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=None)
    kf.get_n_splits()

    print(kf)
    print(kf.get_n_splits())

    # Evenly Distributing Classes: Stratified K-fold
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # skf.get_n_splits(X, y)
    # print(skf)

    # SVM classifier definition
    regressor = linear_model.Lars(fit_intercept=True, verbose=False, normalize=True,
                                  precompute='auto', n_nonzero_coefs=500, eps=2.2204460492503131e-16,
                                  copy_X=True, fit_path=True, positive=False)


    # k-Fold loop
    counter = 0
    model = []
    r2 = []
    print("\nShape")
    print(X.shape)
    print(y.shape)
    for train_index, test_index in kf.split(X, y=y):
        # print("TRAIN: ", train_index, " TEST: ", test_index)
        # get indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ### SVM CLASSIFIER ####################################
        # fit model
        model.append(regressor.fit(X_train, y_train))

        # predict
        y_pred = model[counter].predict(X_test)

        print("Coefficient of Determination for Regressor ", counter)
        r2.append(model[counter].score(X_test, y_test))
        print(r2[counter])

        counter += 1

    # choose svm with highest r2
    idx = np.argmax(r2)
    return model[idx]