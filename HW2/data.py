import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data(N):

    m01 = np.array([-1, -1])
    m02 = np.array([1, 1])
    m11 = np.array([-1, 1])
    m12 = np.array([1, -1])
    c = np.array([[1, 0],
                  [0, 1]])

    label = []
    random.seed(4) # fix a random seed to prove the same random value in each runnning
    one = 0
    zero = 0
    for i in range(N):
        rd = random.random()  #generate random value from 0-1
        #rd = random.uniform(0,1)  # generate random value from 0-1
        if(rd <= 0.6):
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
            if(random.uniform(0, 1) >= 0.5):
                data = np.random.multivariate_normal(m11, c)
            else:
                data = np.random.multivariate_normal(m12, c)
            X.append(data.tolist())
        elif(l == 0):
            if(random.uniform(0, 1) >=0.5):
                data = np.random.multivariate_normal(m01, c)
            else:
                data = np.random.multivariate_normal(m02, c)
            X.append(data.tolist())
    X = np.array(X)
    label = np.array(label)
    #print("X:",X)
    print("shape of X: ", X.shape)
    print("shape of Label: ", label.shape)

    return X,label

def plot_data_2D(X, label):
    # print the first dot with graph label legend
    plt.scatter(X[6][0], X[6][1], color='skyblue', marker='o', label='L=1')
    plt.scatter(X[0][0], X[0][1], color='darkorange', marker='+', label='L=0')

    for i in range(label.size):
        if(label[i] == 1):
            plt.scatter(X[i][0], X[i][1], color='skyblue', marker='o')
        else:
            plt.scatter(X[i][0], X[i][1], color='darkorange', marker='+')
    plt.title("Data distribution " + str(X.shape[0]) + " samples")
    plt.legend()
    plt.show()

def plot_data_distribution():
    X20, label20 = generate_data(20)
    X200, label200 = generate_data(200)
    X2k, label2k = generate_data(2000)
    X10k, label10k = generate_data(10000)

    plot_data_2D(X20, label20)
    plot_data_2D(X200, label200)
    plot_data_2D(X2k, label2k)
    plot_data_2D(X10k, label10k)



def hw2q2():
    Ntrain = 100
    data = generateDataQ2(Ntrain)
    # plot3D(data[0, :], data[1, :], data[2, :])
    xTrain = data[0:2, :]
    yTrain = data[2, :]

    Ntrain = 1000
    data = generateDataQ2(Ntrain)
    # plot3D(data[0, :], data[1, :], data[2, :])
    xValidate = data[0:2, :]
    yValidate = data[2, :]

    return xTrain, yTrain, xValidate, yValidate


def generateDataQ2(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]  # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x, labels = generateDataFromGMM(N, gmmParameters)
    return x


def generateDataFromGMM(N, gmmParameters):
    #    Generates N vector samples from the specified mixture of Gaussians
    #    Returns samples and their component labels
    #    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']  # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    x = np.zeros((n, N))
    labels = np.zeros((1, N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        x[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))

    return x, labels


def plot3D(a, b, c, mark="o", col="b"):
    from matplotlib import pyplot
    # import pylab
    # from mpl_toolkits.mplot3d import Axes3D
    # pylab.ion()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title('Training Dataset')
    plt.legend()
    plt.show()

