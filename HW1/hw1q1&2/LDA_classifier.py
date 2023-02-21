import numpy as np

def lda_classifier(X, label, tau):
    w = cal_w(X, label)
    prediction = []
    for i in range(label.size):
        if(np.dot(w.transpose(), X[i]) <= tau ):
            prediction.append(1)
        else:
            prediction.append(0)
    return np.array(prediction)



def cal_w (X, label):
    data_0, data_1 = [], []
    for i in range(label.size):
        if(label[i] == 1):
            data_1.append(X[i])
        else:
            data_0.append(X[i])
    data_0 = np.matrix(data_0)
    data_1 = np.matrix(data_1)

    m0 = np.mean(data_0, axis=0).transpose()
    m1 = np.mean(data_1, axis=0).transpose()
    # print("mean x of label with 0\n", m0)
    # print("mean x of label with 1\n", m1)
    cov0 = np.cov(data_0.transpose())
    cov1 = np.cov(data_1.transpose())
    # print("cov0\n", cov0)   #4x4
    # print("cov1\n", cov1)   #4x4

    Sw = np.matrix(cov0 + cov1)
    # print("\nSw\n", Sw)
    Sb = np.dot((m0 - m1), (m0 - m1).transpose())
    # print("Sb\n", Sb)
    w = np.dot(Sw.I,m0 - m1)
    # print("w\n", w)

    tao = (data_0.size*np.dot(w.transpose(), m0) + data_1.size*np.dot(w.transpose(), m1))
    tao = tao/(data_0.size + data_1.size)
    #print("tao: ", tao)

    #lin's w
    # SwISb = np.dot(Sw.I, Sb)
    # eig = np.linalg.eig(SwISb)
    #
    # for i in range(eig[0].size):
    #     if eig[0][i] == max(eig[0]):
    #         w = (eig[1].transpose()[i]).transpose()
    # #         print("w_lin ", w)
    return w
