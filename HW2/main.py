from data import *
from ROC import *
from classifier import linear_regression
from classifier import quadratic_regression
from vehicle import *


# # generated data distribution
# plot_data_distribution()


# question1 part 1
# X, label = generate_data(10000)
# plot_ROC_ERM(X, label)

# question1 part 2

## linear

# # train in 20 samples
# train20X, train20Y = generate_data(20)
# testX, testY = generate_data(10000)
# linear_regression(train20X, train20Y, testX, testY)

# # train in 200 samples
# train200X, train200Y = generate_data(200)
# testX, testY = generate_data(10000)
# linear_regression(train200X, train200Y, testX, testY)

# # train in 2k samples
# train2kX, train2kY = generate_data(2000)
# testX, testY = generate_data(10000)
# linear_regression(train2kX, train2kY, testX, testY)


## quadratic regression

# # train in 20 samples
# train20X, train20Y = generate_data(20)
# testX, testY = generate_data(10000)
# quadratic_regression(train20X, train20Y, testX, testY)

# # train in 200 samples
# train200X, train200Y = generate_data(200)
# testX, testY = generate_data(10000)
# quadratic_regression(train200X, train200Y, testX, testY)

# # train in 2k samples
# train2kX, train2kY = generate_data(2000)
# testX, testY = generate_data(10000)
# quadratic_regression(train2kX, train2kY, testX, testY)


# question 2
cubic_polynomial()



# # question 3
# plot_pos_vehicle(1)
# plot_pos_vehicle(2)
# plot_pos_vehicle(3)
# plot_pos_vehicle(4)









