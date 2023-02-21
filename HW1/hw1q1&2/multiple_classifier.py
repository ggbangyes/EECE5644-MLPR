import numpy as np
from ERM_classifier import cal_pdf
from tqdm import tqdm

def map_classifier(X, label):
    p1, p2, p3 = 0.3, 0.3, 0.4  # priori of three classes

    m1 = np.matrix([-3, 6, -1]).transpose()
    m2 = np.matrix([2, 2, 1]).transpose()
    m3 = np.matrix([-4, 1, 7]).transpose()
    m4 = np.matrix([1, 2, 9]).transpose()
    cov = np.matrix([[3, 0, 0],
                    [0, 4, 0],
                    [0, 0, 1]])

    prediction = []

    for i in range(X.shape[0]):
        ph1 = cal_pdf(X[i], m1, cov) * p1
        ph2 = cal_pdf(X[i], m2, cov) * p2
        ph3 = 0.5 * (cal_pdf(X[i], m3, cov) + cal_pdf(X[i], m4, cov))

        if(ph1 >= ph2):
            if(ph1 >= ph3):   # compare 1 and 3
                prediction.append(1)
            else:
                prediction.append(3)
        else:                 # comepare 2 and 3
            if (ph2 >= ph3):
                prediction.append(2)
            else:
                prediction.append(3)

    # print("prediction: ", prediction[0:20])
    # print("label: ", label[0:20])
    return np.array(prediction)


def erm_classifier_lossMatrix(X, loss):
    p1, p2, p3 = 0.3, 0.3, 0.4  # priori of three classes

    m1 = np.matrix([-3, 6, -1]).transpose()
    m2 = np.matrix([2, 2, 1]).transpose()
    m3 = np.matrix([-4, 1, 7]).transpose()
    m4 = np.matrix([1, 2, 9]).transpose()
    cov = np.matrix([[3, 0, 0],
                     [0, 4, 0],
                     [0, 0, 1]])

    prediction, risk = [], []

    for i in tqdm(range(X.shape[0])):
        px = p1*cal_pdf(X[i], m1, cov) + p2*cal_pdf(X[i], m2, cov) + \
             0.5*p3*(cal_pdf(X[i], m3, cov) + cal_pdf(X[i], m4, cov))

        ph1 = (-1)*(loss[0][0]*cal_pdf(X[i], m1, cov)*p1 + \
                    loss[0][1]*cal_pdf(X[i], m2, cov)*p2 + \
                    loss[0][2]*0.5*(cal_pdf(X[i], m3, cov) + cal_pdf(X[i], m4, cov)))

        ph2 = (-1) * (loss[1][0] * cal_pdf(X[i], m1, cov) * p1 + \
                      loss[1][1] * cal_pdf(X[i], m2, cov) * p2 + \
                      loss[1][2] * 0.5 * (cal_pdf(X[i], m3, cov) + cal_pdf(X[i], m4, cov)))

        ph3 = (-1) * (loss[2][0] * cal_pdf(X[i], m1, cov) * p1 + \
                      loss[2][1] * cal_pdf(X[i], m2, cov) * p2 + \
                      loss[2][2] * 0.5 * (cal_pdf(X[i], m3, cov) + cal_pdf(X[i], m4, cov)))

        if (ph1 >= ph2):
            if (ph1 >= ph3):  # compare 1 and 3
                prediction.append(1)
                max_ph = ph1
            else:
                prediction.append(3)
                max_ph = ph3
        else:  # comepare 2 and 3
            if (ph2 >= ph3):
                prediction.append(2)
                max_ph = ph2
            else:
                prediction.append(3)
                max_ph = ph3

        risk.append((-1)*max_ph/px)

    # print("prediction: ", prediction[0:20])
    # print("mean risk: ", np.mean(risk))

    return np.array(prediction), np.mean(risk)














