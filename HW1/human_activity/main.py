from dataPreprocessing import *
from Classifier import *
from eveluation import *

loc_features = 'UCI HAR Dataset/features.txt'
loc_Xtrain = 'UCI HAR Dataset/train/X_train.txt'
loc_Ytrain = 'UCI HAR Dataset/train/y_train.txt'
loc_Xtest = 'UCI HAR dataset/test/X_test.txt'
loc_Ytest = 'UCI HAR dataset/test/y_test.txt'
loc_data = 'UCI HAR Dataset/csv_files/X_label_dataset.csv'

loc_prediction_Reg = 'UCI HAR Dataset/csv_files/prediction_Regul.csv'

loc_prediction_NoReg = 'UCI HAR Dataset/csv_files/prediction_NoRegul.csv'

# data preprocessing, transfer txt to csv file
# txt_to_csv(loc_features, loc_Xtrain, loc_Ytrain, loc_Xtest, loc_Ytest)

# read X and label from csv file
X, label, df = readData(loc_data)

# plot data_distribution and feature distribution
# plot_data_distribution(X,label)
# plot_features_distribution(X, label, df)

# gaussian classifier
gaussian_map_classifier(X, label)

# classify the sample
prediction = readPrediction(loc_prediction_Reg)

# # xgb classifier
# prediction = xgb_classifier(X, label)


# #display the confusion matrix
# display_confusion_matrix(X, label, prediction)
