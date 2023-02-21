import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing



# 1: 'WALKING', 2:'WALKING_UPSTAIRS', 3:'WALKING_DOWNSTAIRS',
# 4:'SITTING', 5:'STANDING', 6:'LAYING'


def txt_to_csv(loc_features, loc_Xtrain, loc_Ytrain, loc_Xtest, loc_Ytest):
    features = []
    with open(loc_features) as f:
        for line in f.readlines():
            features.append(line.split()[1])
    #print(features)
    print('Number of features: ',len(features))

    train_X = pd.read_csv(loc_Xtrain, delim_whitespace=True, header=None)
    test_X = pd.read_csv(loc_Xtest, delim_whitespace=True, header=None)
    X = pd.concat([train_X, test_X], axis=0)
    X.columns = features
    print(X.head())

    train_Y = pd.read_csv(loc_Ytrain, names=['Activity'])
    test_Y = pd.read_csv(loc_Ytest, names=['Activity'])
    label = pd.concat([train_Y, test_Y], axis=0)
    print(label.head(20))


    # #put all X, Y into a dataframe
    dataset = pd.concat([X,label], axis=1)
    print(dataset.head())

    dataset.to_csv('UCI HAR Dataset/csv_files/X_label_dataset.csv', index=False)


def readData(loc_data):
    df = pd.read_csv(loc_data, header=0)

    X = df.iloc[:, 0:561]
    label = df.iloc[:, 561]

    #X = (X-X.mean()) / X.std()
    X = (X - X.min()) / (X.max() - X.min())

    print("X.shape ", X.shape)
    print("label.shape ", label.shape)
    X = np.array(X)
    label = np.array(label)
    return X, label, df

def readPrediction(loc_prediction):
    df = pd.read_csv(loc_prediction)
    print("shape of prediction: ", np.array(df.iloc[:, 0]).shape)
    return np.array(df.iloc[:, 0])


def plot_data_distribution(X,label):
    uniLabel = np.unique(label)
    print("uniLabel: ", uniLabel)

    pca = PCA(n_components=2)
    newX = pca.fit_transform(X)

    plt.scatter(newX[0][0], newX[0][1], color='skyblue', marker='o', label='L=WALKING')
    plt.scatter(newX[0][0], newX[0][1], color='darkorange', marker='+', label='L=WALKING_UPSTAIRS')
    plt.scatter(newX[0][0], newX[0][1], color='green', marker='h', label='L=WALKING_DOWNSTAIRS')
    plt.scatter(newX[0][0], newX[0][1], color='purple', marker='d', label='L=SITTING')
    plt.scatter(newX[0][0], newX[0][1], color='red', marker='^', label='L=STANDING')
    plt.scatter(newX[0][0], newX[0][1], color='yellow', marker='v', label='L=LAYING')

    for i in tqdm(range(X.shape[0])):
        if(label[i] == 1):
            plt.scatter(newX[i][0], newX[i][1], color='skyblue', marker='o')
        elif(label[i] == 2):
            plt.scatter(newX[i][0], newX[i][1], color='darkorange', marker='+')
        elif (label[i] == 3):
            plt.scatter(newX[i][0], newX[i][1], color='green', marker='h')
        elif (label[i] == 4):
            plt.scatter(newX[i][0], newX[i][1], color='purple', marker='d')
        elif (label[i] == 5):
            plt.scatter(newX[i][0], newX[i][1], color='red', marker='^')
        elif (label[i] == 6):
            plt.scatter(newX[i][0], newX[i][1], color='yellow', marker='v')

    plt.title("Data distribution (reduce dimension from 561D to 2D)")
    plt.legend()
    plt.show()


def plot_features_distribution(X, label, df):

    # print(df.columns)
    for j in range(0,6,2):

        plt.scatter(X[0][0], X[0][1], color='skyblue', marker='o', label='L=WALKING')
        plt.scatter(X[0][0], X[0][1], color='darkorange', marker='+', label='L=WALKING_UPSTAIRS')
        plt.scatter(X[0][0], X[0][1], color='green', marker='h', label='L=WALKING_DOWNSTAIRS')
        plt.scatter(X[0][0], X[0][1], color='purple', marker='d', label='L=SITTING')
        plt.scatter(X[0][0], X[0][1], color='red', marker='^', label='L=STANDING')
        plt.scatter(X[0][0], X[0][1], color='yellow', marker='v', label='L=LAYING')


        for i in tqdm(range(X.shape[0])):
            if (label[i] == 1):
                plt.scatter(X[i][j], X[i][j+1], color='skyblue', marker='o')
            elif (label[i] == 2):
                plt.scatter(X[i][j], X[i][j+1], color='darkorange', marker='+')
            elif (label[i] == 3):
                plt.scatter(X[i][j], X[i][j+1], color='green', marker='h')
            elif (label[i] == 4):
                plt.scatter(X[i][j], X[i][j+1], color='purple', marker='d')
            elif (label[i] == 5):
                plt.scatter(X[i][j], X[i][j+1], color='red', marker='^')
            elif (label[i] == 6):
                plt.scatter(X[i][j], X[i][j+1], color='yellow', marker='v')


        plt.title("Data projection of "+ df.columns[j] +" and "+ df.columns[j+1])
        plt.legend()
        plt.show()

