__author__ = 'omohamme'
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import pylab as py
from matplotlib import cm
from matplotlib.colors import LogNorm


X = []
Y = []
Z = []
# with open("all_outputs_2d_V2Controller_good_BruteForce.txt", "r") as file:
with open("all_outputs_2d_V2Controller_damage1.txt", "r") as file:
# with open("all_outputs_2d_V3Controller.txt", "r") as file:
# with open("all_outputs_2d_good.txt", "r") as file:
    file_lines = file.read().splitlines()
    print len(file_lines)
    random.shuffle(file_lines)
    num_points = 1000
    random_indices = random.sample(range(0, len(file_lines)), num_points)
    for i in range(num_points):
        # Get a random index
        random_index = random_indices[i]
        # Get the values
        line = file_lines[random_index]
        input_points = line.split("--->")[0].split(",")
        X.append(map(float, input_points)[0])
        Y.append(map(float, input_points)[1])
        output_points = line.split("--->")[1].replace(" ", "").split(",")
        Z.append(1 + map(float, output_points)[1])# Get the target we want and deduce the mean from it
        # Z.append(-map(float, output_points)[0])# Get the target we want and deduce the mean from it

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
ax.set_xlabel('Amplitude')
# ax.set_xlabel('X 1')
ax.set_ylabel('Phase shift')
# ax.set_ylabel('X 2')
# ax.set_zlabel('Robot speed')
ax.set_zlabel('Height_variance')
plt.show()
# data = [X, Y, Z]
# X = np.asarray(X)
# Y = np.asarray(Y)
# Z = np.asarray(Z)
# plt.scatter(X,Y,c=Z)
# plt.imshow(data)
# plt.colorbar()
# plt.show()