__author__ = 'omohamme'
import matplotlib.pyplot as plt
import numpy as np

gp_data = open("wawa.txt", "r")
gp_lines = gp_data.readlines()
gp_data.close()
mu = []
mu_p_sigma = []
mu_n_sigma = []

mu_obj_0 = []
mu_obj_1 = []
sigma_obj_0 = []
sigma_obj_1 = []
obj_num = 0
read_flag = False
for line in gp_lines:
    # line_split = line.split(" ")
    # mu.append(line_split[1])
    # mu_p_sigma.append(line_split[2])
    # mu_n_sigma.append(line_split[3])
    line = line[:-1]
    if "################START the model points" in line:
        read_flag = True
    elif "################END the model points" in line:
        read_flag = False
    else:
        if read_flag == True:
            if obj_num == 0:
                mu_obj_0.append(-float(line.split(" ")[0]))
                sigma_obj_0.append(float(line.split(" ")[1]))
                obj_num += 1
            elif obj_num == 1:
                mu_obj_1.append(-float(line.split(" ")[0]))
                sigma_obj_1.append(float(line.split(" ")[1]))
                obj_num = 0


mu_obj_0 = np.array(mu_obj_0)
sigma_obj_0 = np.array(sigma_obj_0)
mu_obj_1 = np.array(mu_obj_1)
sigma_obj_1 = np.array(sigma_obj_1)

plt.figure()
plt.plot(mu_obj_0, mu_obj_1, "ro")
plt.show()
# plt.figure()
# # plt.plot(mu, marker='o', linestyle='--')
# plt.plot(mu, linestyle='--')
# plt.plot(mu_p_sigma, linestyle='--')
# plt.plot(mu_n_sigma, linestyle='--')
# plt.legend(["mu", "mu+sigma", "mu-sigma"])
#
# plt.ylabel('Height Variance')
# plt.xlabel('Velocity')
# plt.show()
