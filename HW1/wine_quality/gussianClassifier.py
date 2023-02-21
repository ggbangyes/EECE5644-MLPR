import numpy as np


def gussian_classifier(X, label):
    priori = []
    data_3, data_4, data_5, data_6, data_7, data_8, data_9 = [],[],[],[],[],[],[]
    prediction = []
    uniLabel = np.unique(label)  #3, 4, 5, ... 9
    for l in uniLabel:
        priori.append(np.sum(label == l))
    priori = np.array(priori)/X.shape[0]
    print("priori of quality from 3 to 9", priori)

    for i in range(X.shape[0]):
        if(label[i] == 3):
            data_3.append(X[i])
        elif (label[i] == 4):
            data_4.append(X[i])
        elif (label[i] == 5):
            data_5.append(X[i])
        elif (label[i] == 6):
            data_6.append(X[i])
        elif (label[i] == 7):
            data_7.append(X[i])
        elif (label[i] == 8):
            data_8.append(X[i])
        elif (label[i] == 9):
            data_9.append(X[i])

    data_3 = np.matrix(data_3)
    data_4 = np.matrix(data_4)
    data_5 = np.matrix(data_5)
    data_6 = np.matrix(data_6)
    data_7 = np.matrix(data_7)
    data_8 = np.matrix(data_8)
    data_9 = np.matrix(data_9)

    m3 = np.matrix(np.mean(data_3, axis=0).transpose())
    m4 = np.matrix(np.mean(data_4, axis=0).transpose())
    m5 = np.matrix(np.mean(data_5, axis=0).transpose())
    m6 = np.matrix(np.mean(data_6, axis=0).transpose())
    m7 = np.matrix(np.mean(data_7, axis=0).transpose())
    m8 = np.matrix(np.mean(data_8, axis=0).transpose())
    m9 = np.matrix(np.mean(data_9, axis=0).transpose())

    c3 = np.matrix(np.cov(data_3.transpose()))
    c4 = np.matrix(np.cov(data_4.transpose()))
    c5 = np.matrix(np.cov(data_5.transpose()))
    c6 = np.matrix(np.cov(data_6.transpose()))
    c7 = np.matrix(np.cov(data_7.transpose()))
    c8 = np.matrix(np.cov(data_8.transpose()))
    c9 = np.matrix(np.cov(data_9.transpose()))

    # print("shape of cov label 3 ", c3.shape)

    # # regularization lc = c + lamda*I   lamda = alpha*(trace/rank)
    # alpha = 0.5
    # c3 = c3 + np.ones((11, 11)) *(np.sum(np.diagonal(c3)) / np.linalg.matrix_rank(c3)) * alpha
    # c4 = c4 + np.ones((11, 11)) *(np.sum(np.diagonal(c4)) / np.linalg.matrix_rank(c4)) * alpha
    # c5 = c5 + np.ones((11, 11)) *(np.sum(np.diagonal(c5)) / np.linalg.matrix_rank(c5)) * alpha
    # c6 = c6 + np.ones((11, 11)) *(np.sum(np.diagonal(c6)) / np.linalg.matrix_rank(c6)) * alpha
    # c7 = c7 + np.ones((11, 11)) *(np.sum(np.diagonal(c7)) / np.linalg.matrix_rank(c7)) * alpha
    # c8 = c8 + np.ones((11, 11)) *(np.sum(np.diagonal(c8)) / np.linalg.matrix_rank(c8)) * alpha
    # c9 = c9 + np.ones((11, 11)) *(np.sum(np.diagonal(c9)) / np.linalg.matrix_rank(c9)) * alpha

    #calputer posterio
    for i in range(X.shape[0]):
        ph3 = cal_gussian_pdf(X[i], m3, c3) * priori[0]
        ph4 = cal_gussian_pdf(X[i], m4, c4) * priori[1]
        ph5 = cal_gussian_pdf(X[i], m5, c5) * priori[2]
        ph6 = cal_gussian_pdf(X[i], m6, c6) * priori[3]
        ph7 = cal_gussian_pdf(X[i], m7, c7) * priori[4]
        ph8 = cal_gussian_pdf(X[i], m8, c8) * priori[5]
        ph9 = cal_gussian_pdf(X[i], m9, c9) * priori[6]

        # print(ph3)
        # print(ph4)
        # print(ph5)
        # print(ph6)
        # print(ph7)
        # print(ph8)
        # print(ph9)
        # print("*****************************")

        map = max(ph3, ph4, ph5, ph6, ph7, ph8, ph9)

        if(map == ph3):
            prediction.append(3)
        elif(map == ph4):
            prediction.append(4)
        elif (map == ph5):
            prediction.append(5)
        elif (map == ph6):
            prediction.append(6)
        elif (map == ph7):
            prediction.append(7)
        elif (map == ph8):
            prediction.append(8)
        elif (map == ph9):
            prediction.append(9)

    print("first 20 prediction\n", prediction[0:20])
    print("len of prediction: ", len(prediction))
    return np.array(prediction)



def cal_gussian_pdf(data, m, c):
    data = np.matrix(data).transpose()
    pdf = 1 / (pow(2*np.pi, data.size/2) * pow(np.linalg.det(c), 0.5))
    pdf = pdf * np.exp(-0.5*np.dot( np.dot( (data-m).transpose(), c.I), (data-m)))
    return np.float64(pdf)