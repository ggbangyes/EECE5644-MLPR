import numpy as np
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data(N):
    m0 = (-0.5, -0.5, -0.5, -0.5)
    m1 = (1, 1, 1, 1)
    c0 = 0.25 * np.array([[2, -0.5, 0.3, 0],
                          [-0.5, 1, -0.5, 0],
                          [0.3, -0.5, 1, 0],
                          [0, 0, 0, 2]])
    c1 = np.array([[1, 0.3, -0.2, 0],
                    [0.3, 2, 0.3, 0],
                    [-0.2, 0.3, 1, 0],
                    [0, 0, 0, 3]])
    #print(c1)
    label = []
    random.seed(4) # fix a random seed to prove the same random value in each runnning
    one = 0
    zero = 0
    for i in range(N):
        rd = random.random()  #generate random value from 0-1
        #rd = random.uniform(0,1)  # generate random value from 0-1
        if(rd <= 0.35):
            label.append(0)
            zero = zero + 1
        else:
            label.append(1)
            one = one +1
    # print("label", label)
    # print("number of label one: ", one)
    # print("number of label zero: ", zero)

    X = []
    for l in label:
        if(l == 1):
            data = np.random.multivariate_normal(m1, c1)
            X.append(data.tolist())
        elif(l == 0):
            data = np.random.multivariate_normal(m0, c0)
            X.append(data.tolist())
    X = np.array(X)
    label = np.array(label)
    #print("X:",X)
    print("shape of X: ", X.shape)
    print("shape of Label: ", label.shape)

    return X,label


def plot_data(X, label):
    pca = PCA(n_components=2) #tansform 4D data to 2D data
    newX = pca.fit_transform(X)
    #print(newX[0])
    # print the first dot with graph label legend
    plt.scatter(newX[6][0], newX[6][1], color='skyblue', marker='o', label='L=1')
    plt.scatter(newX[0][0], newX[0][1], color='darkorange', marker='+', label='L=0')

    for i in range(label.size):
        if(label[i] == 1):
            plt.scatter(newX[i][0], newX[i][1], color='skyblue', marker='o')
        else:
            plt.scatter(newX[i][0], newX[i][1], color='darkorange', marker='+')
    plt.title("Data distribution (reduce dimension from 4D to 2D)")
    plt.legend()
    plt.show()


def generate_data_3D(N):
    m1 = (-3, 6, -1)
    m2 = (2, 2, 1)
    m3 = (-4, 1, 7)
    m4 = (1, 2, 9)
    cov = np.array([[3, 0, 0],
                   [0, 4, 0],
                   [0, 0, 1]])

    label = []
    random.seed(4)  # fix a random seed to prove the same random value in each runnning
    one, two, three = 0, 0, 0
    for i in range(N):
        rd = random.random()  # generate random value from 0-1
        # rd = random.uniform(0,1)  # generate random value from 0-1
        if (rd <= 0.3):
            label.append(1)
            one +=1
        elif(rd >0.3 and rd<=0.6):
            label.append(2)
            two +=1
        else:
            label.append(3)
            three+=1
    # print("number of 1: ", one)
    # print("number of 2: ", two)
    # print("number of 3:: ", three)

    X = []
    for l in label:
        if (l == 1):
            data = np.random.multivariate_normal(m1, cov)
            X.append(data.tolist())
        elif (l == 2):
            data = np.random.multivariate_normal(m2, cov)
            X.append(data.tolist())
        else:
            if(random.random() <= 0.5):
                data = np.random.multivariate_normal(m3, cov)
                X.append(data.tolist())
            else:
                data = np.random.multivariate_normal(m4, cov)
                X.append(data.tolist())


    X = np.array(X)
    label = np.array(label)
    # print("X:",X)
    # print("label:\n", label[0:20])
    print("shape of X: ", X.shape)
    print("shape of Label: ", label.shape)

    return X, label

def plot_data_3D(X, label):
    figure = plt.figure()
    axes = Axes3D(figure)

    axes.scatter(X[0][0], X[0][1], X[0][2], color='lightcoral', marker='o', label='L=1')
    axes.scatter(X[2][0], X[2][1], X[2][2], color='turquoise', marker='+', label='L=2')
    axes.scatter(X[6][0], X[6][1], X[6][2], color='mediumpurple', marker='2', label='L=3')

    for i in range(0, label.size):
        if label[i] == 1:
            axes.scatter(X[i][0], X[i][1], X[i][2], color='lightcoral', marker='o')
        elif label[i] == 2:
            axes.scatter(X[i][0], X[i][1], X[i][2], color='turquoise', marker='+')
        else:
            axes.scatter(X[i][0], X[i][1], X[i][2], color='mediumpurple', marker='2')

    plt.legend()
    plt.show()