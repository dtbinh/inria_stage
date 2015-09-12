__author__ = 'omohamme'
###########################################################
# The target from this script is to make visual analysis for GP in both
# 1-D and 2-D ONLY. The target is to mainly understand the GP
###########################################################
from gp_robot_speed import robot_speed_gp_excitingData, get_observation_mean, robot_speed_testing_excitingData
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# The design of experiments
dimension = 2
observed_mean = get_observation_mean(observation_num=1)
problem = robot_speed_gp_excitingData(num_in_samples=200, mean_value=observed_mean)
test_methods = robot_speed_testing_excitingData(num_test_samples=1000, mean_value=observed_mean)
X = problem.samples
# x_old = test_methods.test_points
# y_old = test_methods.test_points_results
x = test_methods.test_points
y = test_methods.test_points_results
# print x
# print x_old

# x = []
# y = []
# sorted_index = sorted(range(len(x_old)), key=lambda k: x_old[k])

# for i in sorted_index:
#     x.append(x_old[i])
#     y.append(y_old[i] + observed_mean)

problem.fit_gp()
# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, MSE = problem.gp.predict(x, eval_MSE=True)
y_pred += observed_mean
sigma = np.sqrt(MSE)
fig = plt.figure()

# print X
if dimension == 1:
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    plt.plot(x, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b*', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]), \
            np.concatenate([y_pred - 1.9600 * sigma,
                           (y_pred + 1.9600 * sigma)[::-1]]), \
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$Robot Speed - Mean$')
    plt.ylim(-0.02, 0.02)
    plt.legend(loc='upper left')

    plt.show()
elif dimension == 2:
    ax = fig.add_subplot(111, projection='3d')
    x_new = []
    y_new = []
    z_new = []
    for i in range(len(x)):
        x_new.append(x[i][0])
        y_new.append(x[i][1])

    x_new, y_new = np.meshgrid(x_new, y_new)
    # surf = ax.plot_surface(x_new, y_new, y, 'r.', rstride=1, cstride=1, cmap=cm.coolwarm,
    #     linewidth=0, antialiased=False, label=u'Observations')
    plt.scatter(x_new, y_new, y, c="r", marker="*", label=u'Prediction')
    plt.scatter(x_new, y_new, y_pred, c="b", marker="o", label=u'Observation')
    # plt.fill(np.concatenate([x, x[::-1]]), \
    #         np.concatenate([y_pred - 1.9600 * sigma,
    #                        (y_pred + 1.9600 * sigma)[::-1]]), \
    #         alpha=.5, fc='b', ec='None', label='95% confidence interval')
    # plt.xlabel('$x$')
    # plt.ylabel('$Robot Speed - Mean$')
    # plt.ylim(-0.02, 0.02)
    # plt.legend(loc='upper left')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Parameter 0')
    ax.set_ylabel('Parameter 1')
    ax.set_zlabel('RobotSpeed - Mean')
    plt.show()
else:
    print "Enter a good value dimension value: 1, 2"