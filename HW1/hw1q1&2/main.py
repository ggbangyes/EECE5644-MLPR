from generateData import *
from ROC import *
from confusion_matrix import *
from multiple_classifier import *

N = 10000
# ********************* question 1 *******************
# #plot the data distribution
# X, label = generate_data(10000)
# plot_data(X, label)

# Q1 part A 1.
# X, label = generate_data(N)
# erm_classify(X, p1/p0)

# Q1 part A 2 and 3.
# X, label = generate_data(N)
# plot_ROC_ERM(X, label)

# # Q1 part B.
# X, label = generate_data(N)
# plot_ROC_NBayesian(X, label)

# # Q1 part C.
# X, label = generate_data(N)
# plot_ROC_LDA(X, label)

# ********************* question 2 *******************
# # Q2 part A 1
# X, label = generate_data_3D(N)
# plot_data_3D(X, label)

# # Q2 part A 2
# X, label = generate_data_3D(N)
# confu_matrix(X, label)

# # Q2 part A 3
# X, label = generate_data_3D(N)
# plot_classify_result(X, label, map_classifier(X, label))

# Q2 part B
X, label = generate_data_3D(N)
plot_classify_result_lossMatrix(X, label)
