from sklearn.metrics import confusion_matrix

def display_confusion_matrix(X, label, prediction):

    cm = confusion_matrix(label, prediction)
    print("confusion matrix:\n", cm)

    correct = 0
    for i in range(7):
        for j in range(7):
            if(i == j):
                correct = correct + cm[i][j]

    errorRate = 1 - (correct / len(label))

    print("error rate: ", errorRate)
