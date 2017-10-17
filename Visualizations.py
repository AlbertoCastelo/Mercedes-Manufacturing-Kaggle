import numpy as np
import matplotlib.pyplot as plt


def visio_pca_dim(r2, n_dim):
    x = np.arange(1,n_dim)

    plt.plot(x, r2[0:n_dim], 'bx')
    plt.axis([np.min(x) - 1, np.max(x) + 1, - 0.05, 1 + 0.05])

    plt.xlabel('Number of Dimensions')
    plt.ylabel('Coefficient of Determination R2')
    plt.title('Trade-off Dimensions-Error')

    plt.show()

def visualize_feature_importance(model):
    model.feature_importances_

def visualize_y_categorical(x_train,y_train):
    pass

def visualize_y_ID(x_train,y):
    id = x_train.ix[:,'ID']
    id = id.as_matrix()
    y = y.as_matrix()

    plt.plot(id, y, 'bx')
    plt.axis([np.min(id)-1, np.max(id)+1, np.min(y)-10, np.max(y)+10])

    plt.xlabel('ID')
    plt.ylabel('Y')
    plt.title('Variation of Y with respect to ID')

    plt.show()

def histogram_y(y):
    y = y.as_matrix()

    plt.hist(y, 50, normed=1, facecolor='g', alpha=0.75)
    # plt.axis([np.min(id) - 1, np.max(id) + 1, np.min(y) - 10, np.max(y) + 10])
    plt.xlabel('Y range')
    plt.ylabel('Frequency')
    plt.title('Histogram of Y')
    plt.show()
