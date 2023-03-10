import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import SGDRegressor
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from data import *


def erm_classifier(X, gamma):
    prediction = []
    for i in range(X.shape[0]):
        if(cal_likelihood_ratio(X[i]) >= gamma):
            prediction.append(1)
        else:
            prediction.append(0)
    #print("prediction:\n",prediction)
    return np.array(prediction)

def cal_likelihood_ratio(data):
    m01 = np.matrix([-1, -1]).transpose()
    m02 = np.matrix([1, 1]).transpose()
    m11 = np.matrix([-1, 1]).transpose()
    m12 = np.matrix([1, -1]).transpose()

    c = np.matrix([[1, 0],
                   [0, 1]])

    ratio = (cal_pdf(data, m11, c)+cal_pdf(data, m12, c))/ \
            (cal_pdf(data, m01, c) + cal_pdf(data, m02, c))
    # print("ratio: ", ratio)
    return ratio

def cal_pdf(data, m, c):
    data = np.matrix(data).transpose()
    pdf = 1 / (pow(2*np.pi, data.size/2) * pow(np.linalg.det(c), 0.5))
    pdf = pdf * np.exp(-0.5*np.dot( np.dot( (data-m).transpose(), c.I), (data-m)))
    return np.float64(pdf)

    # sample = np.matrix(data).transpose()
    #
    # g = 1 / (pow(2 * np.pi, sample.size / 2) * pow(np.linalg.det(c), 0.5))
    # g = g * np.exp(-0.5 * np.dot(np.dot((sample - m).transpose(), c.I), (sample - m)))
    # return np.float64(g)


def linear_regression(trianX, trainY, testX, testY):
    model = SGDRegressor()
    model.fit(trianX, trainY)

    print("The intercept term: ", model.intercept_)
    print("Weights assigned to the features: ", model.coef_)

    prediction = model.predict(testX)
    for i in range(prediction.size):
        if(prediction[i]) >=0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0

    print("test prediction\n", prediction[0:20])
    print("test label\n", testY[0:20])
    prediction = np.array(prediction)

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(prediction.size):
        if (prediction[i] == 1 and testY[i] == 1):
            TP = TP + 1
        elif (prediction[i] == 1 and testY[i] == 0):
            FP = FP + 1
        elif (prediction[i] == 0 and testY[i] == 0):
            TN = TN + 1
        elif (prediction[i] == 0 and testY[i] == 1):
            FN = FN + 1

    errorRate = (FP + FN) / prediction.size
    print("error rate: ", errorRate)

    # plot decision boundry
    markers = ('x', '*')
    colors = ('bisque', 'skyblue')
    colors2 = ('darkorange', 'deepskyblue')
    colors3 = ('red','green')
    cmap = ListedColormap(colors[0:len(np.unique(testY))])

    x1_min, x1_max = testX[:, 0].min() - 1, testX[:, 0].max() + 1
    x2_min, x2_max = testX[:, 1].min() - 1, testX[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.05),
                           np.arange(x2_min, x2_max, 0.05))
    Z = model.predict(np.array([xx1.ravel(),xx2.ravel()]).T)

    for i in range(Z.size):
        if(Z[i]) >=0.5:
            Z[i] = 1
        else:
            Z[i] = 0

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    decision =[]
    for i in range(testY.size):
        if (testY[i] == prediction[i]):
            decision.append(1)
        else:
            decision.append(0)


    # plot class samples
    for idx, element in enumerate(np.unique(testY)):
        plt.scatter(x=testX[testY == element, 0],
                    y=testX[testY == element, 1],
                    alpha=0.6,
                    c=colors2[idx],
                    # marker=markers[idx],
                    label="L=" + str(element))

    # decision true or wrong
    for idx,element in enumerate(np.unique(decision)):
        plt.scatter(x=testX[decision==element,0],
            y = testX[decision==element,1],
            alpha=0.6,
            c=colors3[idx],
            marker = markers[idx],
            label="Decision="+str(element))

    plt.legend(loc='upper left')
    plt.title("Data Distribution with Decision"+
              '(train in '+ str(trianX.shape[0])+' samples' +')' )
    plt.show()


def quadratic_regression(trianX, trainY, testX, testY):
    model = Pipeline([
        ("ploy", PolynomialFeatures(degree=2)),
        ("lin_reg", LinearRegression())
    ])

    model.fit(trianX, trainY)

    prediction = model.predict(testX)
    for i in range(prediction.size):
        if (prediction[i]) >= 0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0

    print("test prediction\n", prediction[0:20])
    print("test label\n", testY[0:20])
    prediction = np.array(prediction)

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(prediction.size):
        if (prediction[i] == 1 and testY[i] == 1):
            TP = TP + 1
        elif (prediction[i] == 1 and testY[i] == 0):
            FP = FP + 1
        elif (prediction[i] == 0 and testY[i] == 0):
            TN = TN + 1
        elif (prediction[i] == 0 and testY[i] == 1):
            FN = FN + 1

    errorRate = (FP + FN) / prediction.size
    print("error rate: ", errorRate)

    # plot decision boundry
    markers = ('x', '*')
    colors = ('bisque', 'skyblue')
    colors2 = ('darkorange', 'deepskyblue')
    colors3 = ('red', 'green')
    cmap = ListedColormap(colors[0:len(np.unique(testY))])

    x1_min, x1_max = testX[:,0].min()-1, testX[:,0].max()+1
    x2_min, x2_max = testX[:,1].min()-1, testX[:,1].max()+1
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, int((x1_max-x1_min)*100)).reshape(-1,1),
        np.linspace(x2_min, x2_max, int((x2_max-x2_min)*100)).reshape(-1,1)
    )
    print("xx1.shape", xx1.shape)
    print("xx2.shape", xx2.shape)
    print("xx1.ravel().shape: ", xx1.ravel().shape)
    print("xx2.ravel().shape: ", xx2.ravel().shape)
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()]) #np.c_ 两个矩阵按列方向相加
    # print("np.c_[xx1.ravel(), xx2.ravel()]: ", np.c_[xx1.ravel(), xx2.ravel()])

    for i in range(Z.size):
        if(Z[i]) >=0.5:
            Z[i] = 1
        else:
            Z[i] = 0
    print("Z.shape", Z.shape)
    Z = Z.reshape(xx1.shape)
    print("Z.shape", Z.shape)
    plt.contourf(xx1, xx2, Z, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    decision = []
    for i in range(testY.size):
        if (testY[i] == prediction[i]):
            decision.append(1)
        else:
            decision.append(0)
    decision = np.array(decision)

    # plot class samples
    for idx, element in enumerate(np.unique(testY)):
        plt.scatter(x=testX[testY == element, 0],
                    y=testX[testY == element, 1],
                    alpha=0.6,
                    c=colors2[idx],
                    # marker=markers[idx],
                    label="L=" + str(element))

    # decision true or wrong
    for idx, element in enumerate(np.unique(decision)):
        plt.scatter(x=testX[decision == element, 0],
                    y=testX[decision == element, 1],
                    alpha=0.6,
                    c=colors3[idx],
                    marker=markers[idx],
                    label="Decision=" + str(element))

    plt.legend(loc='upper left')
    plt.title("Data Distribution with Decision" +
              '(train in ' + str(trianX.shape[0]) + ' samples' + ')')
    plt.show()




def cubic_trans(X):
    n = X.shape[1]
    X1 = X[:, 1]
    X2 = X[:, 2]

    ploy_X = np.column_stack((X, X1*X1, X1*X2, X2*X2, X1*X1*X2,
                              X1*X2*X2, X1*X1*X1, X2*X2*X2))
    return ploy_X

def mse_loss(testX, testY, theta):
    ployX = cubic_trans(testX)
    pre = ployX.dot(theta)
    error = testY - pre
    loss = np.mean(np.power(error, 2))
    # print("loss is ", loss)
    return pre, loss


def cal_theta_mle(trainX, trainY):
    ployX = cubic_trans(trainX)  # 100 x 10
    # (X^T*X)^-1 * X^T * y
    theta = np.linalg.inv(ployX.T.dot(ployX)).dot(ployX.T).dot(trainY)
    print("shape of theta_mle", theta.shape)
    return theta


def cal_theta_map(trainX, trainY, gamma):
    ployX = cubic_trans(trainX)  # 100 x 10
    n_col = ployX.shape[1]
    # (X^T*X+gamma*I)^-1 * X^T * y
    theta = np.linalg.inv(ployX.T.dot(ployX) + gamma*np.ones((n_col,n_col)))\
        .dot(ployX.T).dot(trainY)
    # print("shape of theta_map", theta.shape)
    return theta


def plot_pre_and_true(X1, X2, yTrue, yPre):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X1, X2, yTrue, marker='o', color='green', label='True position')
    ax.scatter(X1, X2, yPre, marker='^', color='blue', label='Predicted position')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    plt.legend()
    plt.show()


def plot_loss_gamma(gammas, loss_arr):

    plt.scatter(gammas, loss_arr)
    plt.xlabel('Gamma values')
    plt.ylabel('Mean squared error loss')
    plt.title('Loss and Gamma')
    plt.legend()
    plt.show()

def cubic_polynomial():
    trainX, trainY, testX, testY = hw2q2()
    trainX = trainX.transpose()
    testX = testX.transpose()
    # print(trainX.shape)
    # print(trainY.shape)
    N_train = len(trainY)
    N_test = len(testY)
    trainX = np.column_stack((np.ones(N_train), trainX))
    testX = np.column_stack((np.ones(N_test), testX))

    # # MLE
    # theta_mle = cal_theta_mle(trainX,trainY)
    #
    # print("loss of MLE on the trainset")
    # pre_train = mse_loss(trainX, trainY, theta_mle)
    # print("loss of MLE on the validation set")
    # pre_validation = mse_loss(testX, testY, theta_mle)
    #
    # # plot_pre_and_true(testX[:, 0], testX[:, 1], testY, pre_validation)

    # MAP
    gammas = np.linspace(1e-6, 100, 1000)
    loss_arr = []

    for i in range(gammas.size):
        theta_map = cal_theta_map(trainX, trainY, gammas[i])

        pre_validation, loss = mse_loss(testX, testY, theta_map)
        loss_arr.append(loss)

    loss_arr = np.array(loss_arr)
    plot_loss_gamma(gammas, loss_arr)

    # plot_pre_and_true(testX[:, 0], testX[:, 1], testY, pre_validation)


