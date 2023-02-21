import matplotlib.pyplot as plt
from tqdm import tqdm

import NBayesian_classifier
from ERM_classifier import *
from LDA_classifier import *
from NBayesian_classifier import *
from sklearn.metrics import roc_curve

def plot_ROC_ERM(X, label):
    gamma = np.append(np.arange(0 , 0.02, 0.001), np.arange(0.02, 0.1, 0.01))
    gamma = np.append(gamma, np.arange(0.1,0.7 , 0.01))
    gamma = np.append(gamma, np.arange(1, 10, 1))
    gamma = np.append(gamma, np.arange(10, 100, 10))
    gamma = np.append(gamma, np.arange(100, 1000, 100))
    gamma = np.append(gamma, np.arange(1000, 10000, 1000))
    gamma = np.append(gamma, np.arange(100000, 1000000, 100000))
    gamma = np.append(gamma, np.inf)

    theoratical_gamma = 0.35/0.65
    scatter_x = []
    scatter_y = []
    bestErrorRate = 1
    bestGamma, bestTPR, bestFPR = 0, 0, 0
    # vary gamma to find TPR, FPR set and best gamma with lowest error rate
    for con in tqdm(range(gamma.size)):
        TP, FP, TN, FN = 0, 0, 0, 0

        prediction = erm_classify(X, gamma[con])

        for i in range(X.shape[0]):
            if(prediction[i]==1 and label[i]==1):
                TP = TP + 1
            elif(prediction[i]==1 and label[i]==0):
                FP = FP + 1
            elif (prediction[i] == 0 and label[i] == 0):
                TN = TN + 1
            elif (prediction[i] == 0 and label[i] == 1):
                FN = FN + 1

        TPR = TP / (TP + FN)
        FPR = FP / (TN + FP)
        scatter_y.append(TPR)
        scatter_x.append(FPR)

        errorRate = (FP+FN) / prediction.size
        if(errorRate < bestErrorRate):
            bestErrorRate = errorRate
            bestGamma = gamma[con]
            bestTPR = TPR
            bestFPR = FPR

    print("y: ", scatter_y)
    print("x: ", scatter_x)
    print("bestErrorRate: ", bestErrorRate)
    print("bestGamma: ", bestGamma)
    print("bestTPR: ", bestTPR)
    print("bestFPR: ", bestFPR)

    #theoratical gamma TPR, FPR
    TP, FP, TN, FN = 0, 0, 0, 0
    theoratic_prediction = erm_classify(X, theoratical_gamma)
    for i in range(X.shape[0]):
        if (theoratic_prediction[i] == 1 and label[i] == 1):
            TP = TP + 1
        elif (theoratic_prediction[i] == 1 and label[i] == 0):
            FP = FP + 1
        elif (theoratic_prediction[i] == 0 and label[i] == 0):
            TN = TN + 1
        elif (theoratic_prediction[i] == 0 and label[i] == 1):
            FN = FN + 1

    theoratic_TPR = TP / (TP + FN)
    theoratic_FPR = FP / (TN + FP)
    theoratic_errorRate = (FP+FN) / prediction.size

    print("theoratic_TPR: ", theoratic_TPR)
    print("theoratic_FPR:", theoratic_FPR)
    print("theoratic_errorRate:", theoratic_errorRate)

    # plot ROC curve
    #plt.scatter(scatter_x, scatter_y)
    plt.plot(scatter_x, scatter_y)
    plt.title("The ROC curve of EMR Classifier")
    plt.xlabel("False Positive Rate (FPR); gamma")
    plt.ylabel("True Positive Rate (TPR); gamma")
    #plt.show()

    # best empirically gamma dot determined by lowest p(erorr)
    bestGa =  str(bestGamma).split('.')[0]+'.'+str(bestGamma).split('.')[1][:3]
    plt.scatter(bestFPR, bestTPR, color='red', marker='*', s=80,
                label='gamma='+bestGa+'(epirically)')

    # theoratically gama dot determined by lowest p(erorr)
    theoratical_Ga = str(theoratical_gamma).split('.')[0] + '.' + str(theoratical_gamma).split('.')[1][:3]
    plt.scatter(bestFPR, bestTPR, color='yellowgreen', marker='x', s=80,
                label='gamma=' + theoratical_Ga + '(theratically)')

    plt.legend()
    plt.show()

def plot_ROC_NBayesian(X, label):
    gamma = np.append(np.arange(0 , 0.02, 0.001), np.arange(0.02, 0.1, 0.01))
    gamma = np.append(gamma, np.arange(0.1,0.7 , 0.01))
    gamma = np.append(gamma, np.arange(1, 10, 1))
    gamma = np.append(gamma, np.arange(10, 100, 10))
    gamma = np.append(gamma, np.arange(100, 1000, 100))
    gamma = np.append(gamma, np.arange(1000, 10000, 1000))
    gamma = np.append(gamma, np.arange(100000, 1000000, 100000))
    gamma = np.append(gamma, np.inf)

    theoratical_gamma = 0.35/0.65
    scatter_x = []
    scatter_y = []
    bestErrorRate = 1
    bestGamma, bestTPR, bestFPR = 0, 0, 0
    # vary gamma to find TPR, FPR set and best gamma with lowest error rate
    for con in tqdm(range(gamma.size)):
        TP, FP, TN, FN = 0, 0, 0, 0

        prediction = nb_classify(X, gamma[con])

        for i in range(X.shape[0]):
            if(prediction[i]==1 and label[i]==1):
                TP = TP + 1
            elif(prediction[i]==1 and label[i]==0):
                FP = FP + 1
            elif (prediction[i] == 0 and label[i] == 0):
                TN = TN + 1
            elif (prediction[i] == 0 and label[i] == 1):
                FN = FN + 1

        TPR = TP / (TP + FN)
        FPR = FP / (TN + FP)
        scatter_y.append(TPR)
        scatter_x.append(FPR)

        errorRate = (FP+FN) / prediction.size
        if(errorRate < bestErrorRate):
            bestErrorRate = errorRate
            bestGamma = gamma[con]
            bestTPR = TPR
            bestFPR = FPR

    print("y: ", scatter_y)
    print("x: ", scatter_x)
    print("bestErrorRate: ", bestErrorRate)
    print("bestGamma: ", bestGamma)
    print("bestTPR: ", bestTPR)
    print("bestFPR: ", bestFPR)

    #theoratical gamma TPR, FPR
    TP, FP, TN, FN = 0, 0, 0, 0
    theoratic_prediction = nb_classify(X, theoratical_gamma)
    for i in range(X.shape[0]):
        if (theoratic_prediction[i] == 1 and label[i] == 1):
            TP = TP + 1
        elif (theoratic_prediction[i] == 1 and label[i] == 0):
            FP = FP + 1
        elif (theoratic_prediction[i] == 0 and label[i] == 0):
            TN = TN + 1
        elif (theoratic_prediction[i] == 0 and label[i] == 1):
            FN = FN + 1

    theoratic_TPR = TP / (TP + FN)
    theoratic_FPR = FP / (TN + FP)
    theoratic_errorRate = (FP+FN) / prediction.size

    print("theoratic_TPR: ", theoratic_TPR)
    print("theoratic_FPR:", theoratic_FPR)
    print("theoratic_errorRate:", theoratic_errorRate)

    # plot ROC curve
    #plt.scatter(scatter_x, scatter_y)
    plt.plot(scatter_x, scatter_y)
    plt.title("The ROC curve of EMR Classifier")
    plt.xlabel("False Positive Rate (FPR); gamma")
    plt.ylabel("True Positive Rate (TPR); gamma")
    #plt.show()

    # best empirically gamma dot determined by lowest p(erorr)
    bestGa =  str(bestGamma).split('.')[0]+'.'+str(bestGamma).split('.')[1][:3]
    plt.scatter(bestFPR, bestTPR, color='red', marker='*', s=80,
                label='gamma='+bestGa+'(epirically)')

    # theoratically gama dot determined by lowest p(erorr)
    theoratical_Ga = str(theoratical_gamma).split('.')[0] + '.' + str(theoratical_gamma).split('.')[1][:3]
    plt.scatter(bestFPR, bestTPR, color='yellowgreen', marker='x', s=80,
                label='gamma=' + theoratical_Ga + '(theratically)')

    plt.legend()
    plt.show()

def plot_ROC_LDA(X, label):
    #tau = np.append(np.arange(0 , 0.02, 0.001), np.arange(0.02, 0.1, 0.01))
    tau = np.append(np.arange(-3, -2.5, 0.1), np.arange(-2.5, 0.5, 0.01))
    tau = np.append(tau, np.arange(0.5, 3, 0.1))


    scatter_x = []
    scatter_y = []
    bestErrorRate = 1
    besttau, bestTPR, bestFPR = 0, 0, 0
    # vary tau to find TPR, FPR set and best gamma with lowest error rate
    for con in tqdm(range(tau.size)):
        TP, FP, TN, FN = 0, 0, 0, 0

        prediction = lda_classifier(X, label, tau[con])
        for i in range(X.shape[0]):
            if(prediction[i]==1 and label[i]==1):
                TP = TP + 1
            elif(prediction[i]==1 and label[i]==0):
                FP = FP + 1
            elif (prediction[i] == 0 and label[i] == 0):
                TN = TN + 1
            elif (prediction[i] == 0 and label[i] == 1):
                FN = FN + 1

        TPR = TP / (TP + FN)
        FPR = FP / (TN + FP)
        scatter_y.append(TPR)
        scatter_x.append(FPR)


        errorRate = (FP+FN) / prediction.size
        if(errorRate < bestErrorRate):
            bestErrorRate = errorRate
            besttau = tau[con]
            bestTPR = TPR
            bestFPR = FPR


    print("y: ", scatter_y)
    print("x: ", scatter_x)
    print("bestErrorRate: ", bestErrorRate)
    print("besttau: ", besttau)
    print("bestTPR: ", bestTPR)
    print("bestFPR: ", bestFPR)


    # plot ROC curve
    #plt.scatter(scatter_x, scatter_y)
    plt.plot(scatter_x, scatter_y)
    plt.title("The ROC curve of LDA Classifier")
    plt.xlabel("False Positive Rate (FPR); tau")
    plt.ylabel("True Positive Rate (TPR); tau")
    #plt.show()

    # best empirically tau dot determined by lowest p(erorr)
    besttau =  str(besttau).split('.')[0]+'.'+str(besttau).split('.')[1][:3]
    plt.scatter(bestFPR, bestTPR, color='red', marker='*', s=80,
                label='tau='+besttau+'(epirically)')


    plt.legend()
    plt.show()












