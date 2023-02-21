from readData import *
from visualization import *
from gussianClassifier import *
from confusionMatrix import *

dataLoc = 'data/winequality-white.csv'

X, label, df = read_data(dataLoc)
# plot_data_distribution(X,label)
#plot_features_distribution(X, label, df)

prediction = gussian_classifier(X, label)
display_confusion_matrix(X, label, prediction)




