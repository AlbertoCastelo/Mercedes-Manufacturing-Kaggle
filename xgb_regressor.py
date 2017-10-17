import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import Regressors as reg
import Visualizations as visio
import Preprocessing as prep

from sklearn.externals import joblib
from sklearn import preprocessing
import xgboost as xgb

def main():
    # Load Data
    train_data, test_data = prep.load_data()

    # Encode Categorical Variables
    x_train, y_train, x_test, id_test = prep.encode_categories(train_data, test_data)

    # Decompositions
    pca = PCA(n_components=5)
    ica = FastICA(n_components=5, max_iter=1000)
    tsvd = TruncatedSVD(n_components=5)
    gp = GaussianRandomProjection(n_components=5)
    sp = SparseRandomProjection(n_components=5, dense_output=True)



main()