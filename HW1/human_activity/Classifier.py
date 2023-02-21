import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split


def gaussian_map_classifier(X, label):
    uniLabel = np.unique(label)
    priori =[]
    data_1, data_2, data_3, data_4, data_5, data_6 = [],[],[],[],[],[]
    regulazarion = 0
    prediction = []

    for l in uniLabel:
        priori.append(np.sum(label == l))

    priori = np.array(priori) / X.shape[0]
    print('priori: ', priori)

    for i in range(X.shape[0]):
        if(label[i] == 1):
            data_1.append(X[i])
        elif(label[i] == 2):
            data_2.append(X[i])
        elif (label[i] == 3):
            data_3.append(X[i])
        elif (label[i] == 4):
            data_4.append(X[i])
        elif (label[i] == 5):
            data_5.append(X[i])
        elif (label[i] == 6):
            data_6.append(X[i])


    data_1 = np.matrix(data_1)
    data_2 = np.matrix(data_2)
    data_3 = np.matrix(data_3)
    data_4 = np.matrix(data_4)
    data_5 = np.matrix(data_5)
    data_6 = np.matrix(data_6)

    # calculate mean and covariance for each class
    m1 = np.matrix(np.mean(data_1, axis=0).transpose())
    m2 = np.matrix(np.mean(data_2, axis=0).transpose())
    m3 = np.matrix(np.mean(data_3, axis=0).transpose())
    m4 = np.matrix(np.mean(data_4, axis=0).transpose())
    m5 = np.matrix(np.mean(data_5, axis=0).transpose())
    m6 = np.matrix(np.mean(data_6, axis=0).transpose())

    c1 = np.matrix(np.cov(data_1.transpose()))
    c2 = np.matrix(np.cov(data_2.transpose()))
    c3 = np.matrix(np.cov(data_3.transpose()))
    c4 = np.matrix(np.cov(data_4.transpose()))
    c5 = np.matrix(np.cov(data_5.transpose()))
    c6 = np.matrix(np.cov(data_6.transpose()))

    # regularization
    # regularization lc = c + lamda*I   lamda = alpha*(trace/rank)

    alpha, regulazarion = 0.02, 1  # set for sure whether use regularization term
    c1 = c1 + np.ones((561, 561)) * (np.sum(np.diagonal(c1)) / np.linalg.matrix_rank(c1)) * alpha
    c2 = c2 + np.ones((561, 561)) * (np.sum(np.diagonal(c2)) / np.linalg.matrix_rank(c2)) * alpha
    c3 = c3 + np.ones((561, 561)) * (np.sum(np.diagonal(c3)) / np.linalg.matrix_rank(c3)) * alpha
    c4 = c4 + np.ones((561, 561)) * (np.sum(np.diagonal(c4)) / np.linalg.matrix_rank(c4)) * alpha
    c5 = c5 + np.ones((561, 561)) * (np.sum(np.diagonal(c5)) / np.linalg.matrix_rank(c5)) * alpha
    c6 = c6 + np.ones((561, 561)) * (np.sum(np.diagonal(c6)) / np.linalg.matrix_rank(c6)) * alpha



    #calculate posterio
    for i in tqdm(range(X.shape[0])):
        ph1 = np.log(cal_gaussian_pdf(X[i], m1, c1)) * priori[0]
        ph2 = np.log(cal_gaussian_pdf(X[i], m2, c2)) * priori[1]
        ph3 = np.log(cal_gaussian_pdf(X[i], m3, c3)) * priori[2]
        ph4 = np.log(cal_gaussian_pdf(X[i], m4, c4)) * priori[3]
        ph5 = np.log(cal_gaussian_pdf(X[i], m5, c5)) * priori[4]
        ph6 = np.log(cal_gaussian_pdf(X[i], m6, c6)) * priori[5]

        print(ph1)
        print(ph2)
        print(ph3)
        print(ph4)
        print(ph5)
        print(ph6)
        print("*****************************")

        map = max(ph1, ph2, ph3, ph4, ph5, ph6)

        if(map == ph1):
            prediction.append(1)
        elif(map == ph2):
            prediction.append(2)
        elif (map == ph3):
            prediction.append(3)
        elif (map == ph4):
            prediction.append(4)
        elif (map == ph5):
            prediction.append(5)
        elif (map == ph6):
            prediction.append(6)

    print("first 20 prediction\n", prediction[0:20])

    prediction = np.array(prediction)

    if(regulazarion == 1):
        np.savetxt('UCI HAR Dataset/csv_files/prediction_Regul.csv', prediction, delimiter=",", fmt='%d')
        print("write to prediction_Regul.csv successful!")
    elif(regulazarion == 0):
        np.savetxt('UCI HAR Dataset/csv_files/prediction_NoRegul.csv', prediction, delimiter=",", fmt='%d')
        print("write to prediction_NoRegul.csv successful!")
    else:
        print("fail to write!!!")


def cal_gaussian_pdf(data, m, c):
    data = np.matrix(data).transpose()
    pdf = 1 / (np.power(2*np.pi, data.size/2) * np.power(np.linalg.det(c), 0.5))
    #pdf = np.power(2*np.pi, data.size) * np.power(np.linalg.det(c), -0.5)
    pdf = pdf * np.exp(-0.5*np.dot( np.dot( (data-m).transpose(), c.I), (data-m)))
    return np.float64(pdf)


def xgb_classifier(X, label):
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.5, random_state=0)

    model = xgb.XGBClassifier(max_depth=2, learning_rate=0.3, n_estimators=10)
    model.fit(X_train, y_train)

    prediction = model.predict(X)

    print("prediction.shape: ",prediction.shape)

    return np.array(prediction)