import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def read_data(loc):
    df = pd.read_csv(loc, header=0, sep = ';')#.values
    # print(df.shape)

    X = df.iloc[:,0:11]
    X = (X - X.mean()) / X.std()
    label = df["quality"]

    X = np.array(X)
    label = np.array(label)
    print("X.shape ", X.shape)
    print("label.shape ", label.shape)
    print("label ", label[0:20])

    return X, label, df


def plot_data_distribution(X,label):
    uniLabel = np.unique(label)
    print("uniLabel: ", uniLabel)

    pca = PCA(n_components=2)
    newX = pca.fit_transform(X)

    plt.scatter(newX[0][0], newX[0][1], color='skyblue', marker='o', label='L=3')
    plt.scatter(newX[0][0], newX[0][1], color='darkorange', marker='+', label='L=4')
    plt.scatter(newX[0][0], newX[0][1], color='green', marker='h', label='L=5')
    plt.scatter(newX[0][0], newX[0][1], color='purple', marker='d', label='L=6')
    plt.scatter(newX[0][0], newX[0][1], color='red', marker='^', label='L=7')
    plt.scatter(newX[0][0], newX[0][1], color='yellow', marker='v', label='L=8')
    plt.scatter(newX[0][0], newX[0][1], color='grey', marker='<', label='L=9')

    for i in tqdm(range(X.shape[0])):
        if(label[i] == 3):
            plt.scatter(newX[i][0], newX[i][1], color='skyblue', marker='o')
        elif(label[i] == 4):
            plt.scatter(newX[i][0], newX[i][1], color='darkorange', marker='+')
        elif (label[i] == 5):
            plt.scatter(newX[i][0], newX[i][1], color='green', marker='h')
        elif (label[i] == 6):
            plt.scatter(newX[i][0], newX[i][1], color='purple', marker='d')
        elif (label[i] == 7):
            plt.scatter(newX[i][0], newX[i][1], color='red', marker='^')
        elif (label[i] == 8):
            plt.scatter(newX[i][0], newX[i][1], color='yellow', marker='v')
        elif (label[i] == 9):
            plt.scatter(newX[i][0], newX[i][1], color='grey', marker='<')

    plt.title("Data distribution (reduce dimension from 11D to 2D)")
    plt.legend()
    plt.show()

def plot_features_distribution(X, label, df):

    # print(df.columns)
    for j in range(0,3):

        plt.scatter(X[0][0], X[0][1], color='skyblue', marker='o', label='L=3')
        plt.scatter(X[0][0], X[0][1], color='darkorange', marker='+', label='L=4')
        plt.scatter(X[0][0], X[0][1], color='green', marker='h', label='L=5')
        plt.scatter(X[0][0], X[0][1], color='purple', marker='d', label='L=6')
        plt.scatter(X[0][0], X[0][1], color='red', marker='^', label='L=7')
        plt.scatter(X[0][0], X[0][1], color='yellow', marker='v', label='L=8')
        plt.scatter(X[0][0], X[0][1], color='grey', marker='<', label='L=9')

        for i in tqdm(range(X.shape[0])):
            if (label[i] == 3):
                plt.scatter(X[i][j], X[i][j+1], color='skyblue', marker='o')
            elif (label[i] == 4):
                plt.scatter(X[i][j], X[i][j+1], color='darkorange', marker='+')
            elif (label[i] == 5):
                plt.scatter(X[i][j], X[i][j+1], color='green', marker='h')
            elif (label[i] == 6):
                plt.scatter(X[i][j], X[i][j+1], color='purple', marker='d')
            elif (label[i] == 7):
                plt.scatter(X[i][j], X[i][j+1], color='red', marker='^')
            elif (label[i] == 8):
                plt.scatter(X[i][j], X[i][j+1], color='yellow', marker='v')
            elif (label[i] == 9):
                plt.scatter(X[i][j], X[i][j+1], color='grey', marker='<')

        plt.title("Data projection of feature "+ df.columns[j] +" and feature "+ df.columns[j+1])
        plt.legend()
        plt.show()

