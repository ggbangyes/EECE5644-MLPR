from data import *
from classifier import *


# ***************************************** Queston 1 *****************************************
# # Generate Data
# X_100, label_100 = generate_data(100)
# X_200, label_200 = generate_data(200)
# X_500, label_500 = generate_data(500)
# X_1k, label_1k = generate_data(1000)
# X_2k, label_2k = generate_data(2000)
# X_5k, label_5k = generate_data(5000)
# X_test, Y_test = generate_data(10000)


# # plot data distribution
# plot_data_3D(X_100, label_100)
# plot_data_3D(X_200, label_200)
# plot_data_3D(X_500, label_500)
# plot_data_3D(X_1k, label_1k)
# plot_data_3D(X_2k, label_2k)
# plot_data_3D(X_5k, label_5k)
# plot_data_3D(X_test, Y_test)

# # optimal classifier
# optimal_classifier(X_test, Y_test)

# # model selection
# Xs, Ys = [], []
#
# Xs.append(X_100)
# Xs.append(X_200)
# Xs.append(X_500)
# Xs.append(X_1k)
# Xs.append(X_2k)
# Xs.append(X_5k)
#
# Ys.append(label_100)
# Ys.append(label_200)
# Ys.append(label_500)
# Ys.append(label_1k)
# Ys.append(label_2k)
# Ys.append(label_5k)

# plot_model_selection(Xs, Ys)

# plot_assesment(Xs, Ys, X_test, Y_test)

# ***************************************** Queston 2 *****************************************


X_10, Y_10 = generate_data_2(10)
X_100, Y_100 = generate_data_2(100)
X_1k, Y_1k = generate_data_2(1000)
X_10k, Y_10k = generate_data_2(10000)

# plot_data_2D(X_10, Y_10)
# plot_data_2D(X_100, Y_100)
# plot_data_2D(X_1k, Y_1k)
# plot_data_2D(X_10k, Y_10k)

gmm_classifier(X_10)
gmm_classifier(X_100)
gmm_classifier(X_1k)
gmm_classifier(X_10k)




