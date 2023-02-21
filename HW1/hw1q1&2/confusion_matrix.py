import matplotlib.pyplot as plt
import numpy as np
from multiple_classifier import *
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def confu_matrix(X, label):
    prediction = map_classifier(X, label)
    cm = np.zeros((3,3), dtype=np.int)

    for i in range(prediction.size):
        if(label[i]==1 and prediction[i]==1):
            cm[0][0] += 1
        elif(label[i]==1 and prediction[i]==2):
            cm[0][1] += 1
        elif(label[i]==1 and prediction[i]==3):
            cm[0][2] += 1

        elif(label[i]==2 and prediction[i]==1):
            cm[1][0] += 1
        elif (label[i] == 2 and prediction[i] == 2):
            cm[1][1] += 1
        elif (label[i] == 2 and prediction[i] == 3):
            cm[1][2] += 1

        elif (label[i] == 3 and prediction[i] == 1):
            cm[2][0] += 1
        elif (label[i] == 3 and prediction[i] == 2):
            cm[2][1] += 1
        elif (label[i] == 3 and prediction[i] == 3):
            cm[2][2] += 1

    print("confusion matrix (row represents true label, colounm represents prediction):")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            print(cm[i][j], end=" ")
        print()


def plot_classify_result(X, label, prediction):

    figure = plt.figure()
    axes = Axes3D(figure)

    axes.scatter(X[0][0], X[0][1], X[0][2], color='seagreen', marker='v', alpha=0.6, label='L=1 (correct)')
    axes.scatter(X[0][0], X[0][1], X[0][2], color='red', marker='v', alpha=0.6, label='L=1 (wrong)')
    axes.scatter(X[0][0], X[0][1], X[0][2], color='seagreen', marker='x', alpha=0.6, label='L=2 (correct)')
    axes.scatter(X[0][0], X[0][1], X[0][2], color='red', marker='x', alpha=0.6, label='L=2 (wrong)')
    axes.scatter(X[0][0], X[0][1], X[0][2], color='seagreen', marker='+', alpha=0.6, label='L=3 (correct)')
    axes.scatter(X[0][0], X[0][1], X[0][2], color='red', marker='+', alpha=0.6, label='L=3 (wrong)')

    for i in tqdm(range(label.size)):
        if(label[i]==1 and prediction[i]==1):
            axes.scatter(X[i][0], X[i][1], X[i][2], color='seagreen', marker='v', alpha=0.6)
        elif(label[i]==1 and prediction[i]!= 1):
            axes.scatter(X[i][0], X[i][1], X[i][2], color='red', marker='v', alpha=0.6)

        if(label[i]==2 and prediction[i]==2):
            axes.scatter(X[i][0], X[i][1], X[i][2], color='seagreen', marker='x', alpha=0.6)
        elif(label[i]==2 and prediction[i]!= 2):
            axes.scatter(X[i][0], X[i][1], X[i][2], color='red', marker='x', alpha=0.6)

        if(label[i]==3 and prediction[i]==3):
            axes.scatter(X[i][0], X[i][1], X[i][2], color='seagreen', marker='+', alpha=0.6)
        elif(label[i]==3 and prediction[i]!=3):
            axes.scatter(X[i][0], X[i][1], X[i][2], color='red', marker='+', alpha=0.6)

    plt.legend()
    plt.show()

def plot_classify_result_lossMatrix(X, label):
    loss1 = np.array([[0, 1, 10],
                       [1, 0, 10],
                       [1, 1, 0]])
    loss2 = np.array([[0, 1, 100],
                       [1, 0, 100],
                       [1, 1, 0]])


    prediction, risk = erm_classifier_lossMatrix(X, loss1)
    print("start plotting graph of loss1")
    print("expected risk with loss1: ", risk)
    plot_classify_result(X, label, prediction)

    prediction, risk = erm_classifier_lossMatrix(X, loss2)
    print("start plotting graph of loss2")
    print("expected risk with loss2: ", risk)
    plot_classify_result(X, label, prediction)




