import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data(N):
    # 调整后的分布
    m0 = np.array([-3, 3, -9])
    m1 = np.array([2, 6, 3])
    m2 = np.array([1, -2, 8])
    m3 = np.array([0, -9, -3])

    cov = np.array([[10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]])

    label = []
    random.seed(4) # fix a random seed to prove the same random value in each runnning
    np.random.seed(4)
    for i in range(N):
        rd = random.random()  #generate random value from 0-1
        #rd = random.uniform(0,1)  # generate random value from 0-1
        if(rd <= 0.25):
            label.append(0)
        elif(rd <= 0.5):
            label.append(1)
        elif(rd <= 0.75):
            label.append(2)
        else:
            label.append(3)
    # print("label", label)

    X = []
    for l in label:
        if(l == 0):
            data = np.random.multivariate_normal(m0, cov)
            X.append(data.tolist())
        elif(l == 1):
            data = np.random.multivariate_normal(m1, cov)
            X.append(data.tolist())
        elif (l == 2):
            data = np.random.multivariate_normal(m2, cov)
            X.append(data.tolist())
        else:
            data = np.random.multivariate_normal(m3, cov)
            X.append(data.tolist())

    X = np.array(X)
    label = np.array(label)
    #print("X:",X)
    print("shape of X: ", X.shape)
    print("shape of Label: ", label.shape)

    return X,label

def plot_data_3D(X, label):
    figure = plt.figure()
    axes = Axes3D(figure)

    axes.scatter(X[0][0], X[0][1], X[0][2], color='lightcoral', marker='o', label='L=1')
    axes.scatter(X[0][0], X[0][1], X[0][2], color='turquoise', marker='+', label='L=2')
    axes.scatter(X[0][0], X[0][1], X[0][2], color='mediumpurple', marker='2', label='L=3')
    axes.scatter(X[0][0], X[0][1], X[0][2], color='red', marker='x', label='L=4')

    for i in range(0, label.size):
        if label[i] == 0:
            axes.scatter(X[i][0], X[i][1], X[i][2], color='lightcoral', marker='o')
        elif label[i] == 1:
            axes.scatter(X[i][0], X[i][1], X[i][2], color='turquoise', marker='+')
        elif label[i] == 2:
            axes.scatter(X[i][0], X[i][1], X[i][2], color='mediumpurple', marker='2')
        else:
            axes.scatter(X[i][0], X[i][1], X[i][2], color='red', marker='x')
    plt.legend()
    plt.show()


def generate_data_2(N):
    # 调整后的分布
    m0 = np.array([5, 5])
    m1 = np.array([5, -5])
    m2 = np.array([-5, 5])
    m3 = np.array([-5, -5])

    cov0 = np.array([[6, -1],
                     [-1, 6]])
    cov1 = np.array([[7, 3],
                     [3, 7]])
    cov2 = np.array([[6, 0],
                     [0, 6]])
    cov3 = np.array([[8, -5],
                     [-5, 8]])

    label = []
    random.seed(4) # fix a random seed to prove the same random value in each runnning
    np.random.seed(4)
    for i in range(N):
        rd = random.random()  #generate random value from 0-1
        #rd = random.uniform(0,1)  # generate random value from 0-1
        if(rd <= 0.1):      # 0.1, 0.2, 0.3, 0.4
            label.append(0)
        elif(rd <= 0.3):
            label.append(1)
        elif(rd <= 0.6):
            label.append(2)
        else:
            label.append(3)
    # print("label", label)

    X = []
    for l in label:
        if(l == 0):
            data = np.random.multivariate_normal(m0, cov0)
            X.append(data.tolist())
        elif(l == 1):
            data = np.random.multivariate_normal(m1, cov1)
            X.append(data.tolist())
        elif (l == 2):
            data = np.random.multivariate_normal(m2, cov2)
            X.append(data.tolist())
        else:
            data = np.random.multivariate_normal(m3, cov3)
            X.append(data.tolist())

    X = np.array(X)
    label = np.array(label)
    #print("X:",X)
    print("shape of X: ", X.shape)
    print("shape of Label: ", label.shape)

    return X,label

def plot_data_2D(X, label):

    plt.scatter(X[0][0], X[0][1], color='lightcoral', marker='o', label='L=1')
    plt.scatter(X[0][0], X[0][1], color='turquoise', marker='+', label='L=2')
    plt.scatter(X[0][0], X[0][1], color='mediumpurple', marker='2', label='L=3')
    plt.scatter(X[0][0], X[0][1], color='red', marker='x', label='L=4')

    for i in range(0, label.size):
        if label[i] == 0:
            plt.scatter(X[i][0], X[i][1], color='lightcoral', marker='o')
        elif label[i] == 1:
            plt.scatter(X[i][0], X[i][1], color='turquoise', marker='+')
        elif label[i] == 2:
            plt.scatter(X[i][0], X[i][1], color='mediumpurple', marker='2')
        else:
            plt.scatter(X[i][0], X[i][1], color='red', marker='x')

    plt.title("Data distribution " + str(X.shape[0]) + " samples")
    plt.legend()
    plt.show()