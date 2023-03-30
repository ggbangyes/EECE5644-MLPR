import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import mixture

def cal_error_rate(label, prediction):
    error = 0
    for i in range(len(label)):
        if(label[i] != prediction[i]):
            error = error + 1
    error_rate = error / len(label)
    return error_rate


def cal_pdf(data, m, c):
    data = np.matrix(data).transpose()
    pdf = 1 / (pow(2*np.pi, data.size/2) * pow(np.linalg.det(c), 0.5))
    pdf = pdf * np.exp(-0.5*np.dot( np.dot( (data-m).transpose(), c.I), (data-m)))
    return np.float64(pdf)

def optimal_classifier(X, label):
    # 真实的分布
    m0 = np.matrix([-3, 5, -5]).transpose()
    m1 = np.matrix([3, 4, 3]).transpose()
    m2 = np.matrix([-2, 1, 7]).transpose()
    m3 = np.matrix([3, -9, 1]).transpose()

    cov = np.matrix([[10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]])

    prediction = []
    for i in range(X.shape[0]):
        pdf0 = cal_pdf(X[i], m0, cov)
        pdf1 = cal_pdf(X[i], m1, cov)
        pdf2 = cal_pdf(X[i], m2, cov)
        pdf3 = cal_pdf(X[i], m3, cov)

        pdfs = np.array([pdf0, pdf1, pdf2, pdf3])
        prediction.append(np.argmax(pdfs))

    prediction = np.array(prediction)

    error_rate = cal_error_rate(label, prediction)
    print("optimal error rate: ", error_rate)

def MLP_classifier(X, label):
    avg_error = []
    for num_p in tqdm(range(1,30)):
        classifier = MLPClassifier(hidden_layer_sizes=(num_p,), activation='relu',
                                   solver='sgd', learning_rate_init=0.001, max_iter=3000)
        acc = cross_val_score(classifier, X, label, cv=10, scoring='accuracy')
        avg_error.append(1-acc.mean())
        # print("num_p: ", num_p, " accuracy: ", avg_error)

    error_rate = avg_error
    return error_rate


def plot_model_selection(Xs, Ys):

    N_list = [100, 200, 500, 1000, 2000, 5000]
    p_list = list(range(1,30))
    error_rates = []
    for i in range(len(N_list)): #len(N_list)
        error_rates.append( MLP_classifier(Xs[i], Ys[i]) )
        print("error rate of dataset", i, ": ", error_rates[i])

    best_p = []
    min_error = []
    for i in range(len(N_list)):
        min_error.append( min(error_rates[i]) )
        best_p.append( error_rates[i].index(min(error_rates[i])) )
    # on trainset
    print("best_p: ", best_p) # [27, 25, 7, 20, 16, 23]
    print("min_error: ", min_error) # [0.080, 0.050, 0.036, 0.045, 0.0475, 0.050]

    colors = ['orange','skyblue','green','mediumpurple', 'gold','chocolate']

    for i in range(len(N_list)):
        plt.plot(p_list, error_rates[i], label='trainset='+str(N_list[i]),c=colors[i])

    plt.title('Model selection ')
    plt.xlabel('number of perception')
    plt.ylabel('average error rate ')
    plt.legend()
    plt.show()



def MLP_predict(trainX, trainY, testX, testY, num_p):
    avg_error = []

    classifier = MLPClassifier(hidden_layer_sizes=(num_p,), activation='relu',
                               solver='sgd', learning_rate_init=0.001, max_iter=3000)
    classifier.fit(trainX, trainY)

    predict_proba = classifier.predict_proba(testX)
    prediction = []
    for i in range(testX.shape[0]):
        prediction.append(np.argmax(predict_proba[i]))

    error = 0
    for i in range(testX.shape[0]):
        if(testY[i] != prediction[i]):
            error = error + 1

    error_rate = error / testX.shape[0]
    return error_rate


def plot_assesment(train_Xs, train_Ys, testX, testY):
    num_p = [27, 25, 7, 20, 16, 23]
    error_rates =[] #  [0.0682, 0.061, 0.0577, 0.0595, 0.0575, 0.0539]
    N_list = [100, 200, 500, 1000, 2000, 5000]
    for i in range(len(num_p)):
        error_rate = MLP_predict(train_Xs[i], train_Ys[i], testX, testY, num_p[i])
        error_rates.append(error_rate)

    print("error rate on each dataset: \n", error_rates)
    plt.scatter(N_list, error_rates, label='MLP P(error)', c='red')
    plt.hlines(0.06, N_list[0], N_list[-1], linestyles='dashed')
    plt.title("P(error) on Test Dataset with 10k samples")
    plt.legend()
    plt.show()


def gmm_classifier(X):
    likelihood_select_train = []
    likelihood_select_test = []
    aic_select_train = []
    aic_select_test = []
    bic_select_train = []
    bic_select_test = []

    print(X.shape)
    for i in range(30):
        avg_log_likelihood_train = []
        avg_log_likelihood_test = []

        for components in range(1, 7):
            cv = KFold(n_splits=10, shuffle=True)
            log_train = 0
            log_test = 0


            for train_index, test_index in cv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                # print(X_train.shape,X_test.shape)
                train, test, atrain, atest, btrain, btest = \
                    log_likelihood(X_train, X_test, components)
                log_train += train
                log_test += test


            avg_log_likelihood_train.append((float)(log_train / 10))
            avg_log_likelihood_test.append((float)(log_test / 10))


        likelihood_select_train.append(avg_log_likelihood_train.index(max(avg_log_likelihood_train)) + 1)
        likelihood_select_test.append(avg_log_likelihood_test.index(max(avg_log_likelihood_test)) + 1)


        # draw_scores(avg_log_likelihood_train, avg_log_likelihood_test, X.shape[0])

    # print(likelihood_select_train, likelihood_select_test)
    print(likelihood_select_train.count(1), likelihood_select_train.count(2), likelihood_select_train.count(3),
          likelihood_select_train.count(4), likelihood_select_train.count(5), likelihood_select_train.count(6))
    print(likelihood_select_test.count(1), likelihood_select_test.count(2), likelihood_select_test.count(3),
          likelihood_select_test.count(4), likelihood_select_test.count(5), likelihood_select_test.count(6))



def draw_scores(train_likelihood, test_likelihood, N):
    x = [1, 2, 3, 4, 5, 6]
    colors = ('red', 'skyblue', 'lightcoral', 'mediumpurple')

    fig, ax = plt.subplots(2)
    ax[0].plot(x, train_likelihood, c=colors[0], marker='x')
    ax[0].set_title('average log-likelihoods (Training)')

    ax[1].plot(x, test_likelihood, c=colors[1], marker='x')
    ax[1].set_title('average log-likelihoods (Testing)')


    plt.suptitle('N='+str(N))
    plt.tight_layout()
    plt.show()


def log_likelihood(X_train, X_test, components):
    clst = mixture.GaussianMixture(n_components=components)
    clst.fit(X_train)

    return clst.score(X_train), clst.score(X_test), \
        clst.aic(X_train), clst.aic(X_test), clst.bic(X_train), clst.bic(X_test)




