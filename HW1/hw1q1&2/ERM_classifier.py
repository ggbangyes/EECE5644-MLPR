import numpy as np

def erm_classify(X, gamma):
    prediction = []
    for i in range(X.shape[0]):
        if(cal_likelihood_ratio(X[i]) >= gamma):
            prediction.append(1)
        else:
            prediction.append(0)
    #print("prediction:\n",prediction)
    return np.array(prediction)

def cal_likelihood_ratio(data):
    m0 = np.matrix([-0.5, -0.5, -0.5, -0.5]).transpose()
    m1 = np.matrix([1, 1, 1, 1]).transpose()

    c0 = 0.25 * np.matrix([[2, -0.5, 0.3, 0],
                          [-0.5, 1, -0.5, 0],
                          [0.3, -0.5, 1, 0],
                          [0, 0, 0, 2]])
    c1 = np.matrix([[1, 0.3, -0.2, 0],
                   [0.3, 2, 0.3, 0],
                   [-0.2, 0.3, 1, 0],
                   [0, 0, 0, 3]])

    ratio = cal_pdf(data, m1, c1) / cal_pdf(data, m0, c0)
    return ratio

def cal_pdf(data, m, c):
    data = np.matrix(data).transpose()
    pdf = 1 / (pow(2*np.pi, data.size/2) * pow(np.linalg.det(c), 0.5))
    pdf = pdf * np.exp(-0.5*np.dot( np.dot( (data-m).transpose(), c.I), (data-m)))
    return np.float64(pdf)