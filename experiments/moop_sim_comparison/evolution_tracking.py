__author__ = 'omohamme'
# This algorithm is made in order to track hypervolume perforamce of each algorithm during each iteration.
# We want to know when some algorithm outperform the other algorithms and so on.
from PyGMO import *
import matplotlib.pyplot as plt
import os

def pareto_frontier(Xs, Ys, maxX = False, maxY = False):
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    p_front = [myList[0]]
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

def evolution_tracking(scene, ref_point, num_iteration):
    # Parse the files first. For each iterations, calculate the hypervolume for it.
    directory = "./log_files/" + scene[:-4] + "/"
    algorithms_hypervolume = {}
    # Initialize the hypervolume
    for filename in os.listdir(directory):
        # if ("LOG_" not in filename) and ("nsga2" not in filename) and ("ehvinoprior" not in filename) and ("parego" not in filename):
        if ("LOG_" not in filename) and ("nsga2" not in filename):
            algorithm_type = filename.split("_")[1] + filename.split("_")[4]
            log_file = directory + filename
            with open(log_file, "r") as file:
                log_data = file.read().splitlines()
                log_data.reverse()
                clean_data = []
                read_flag = False
                one_iteration_available = False
                for line in log_data:
                    if line == "##############################################":
                        read_flag = True
                        one_iteration_available = False
                    elif line == "#######################The current pareto points":
                        one_iteration_available = True
                        read_flag = False
                    else:
                        if read_flag == True:
                            data = line.split("---->")#[-1].split(",")[:-1]
                            # This piece of code is really overfitting the reports. I should take care of it soon.
                            # print line
                            if len(data[0].split(",")[:-1]) == 2: # That means that this part has the pareto points.
                                good_point = data[0].split(",")[:-1]
                            else:
                                good_point = data[1].split(",")[:-1]
                            ######################################################################################
                            data_numeric = []

                            data_numeric.append(-float(good_point[0]))  # I added the minus here. Result of my work with Mathew
                            data_numeric.append(-float(good_point[1]))  # I added the minus here. Result of my work with Mathew
                            clean_data.append(data_numeric)
                        elif one_iteration_available:
                            # Draw the hypervolume
                            hv = util.hypervolume(clean_data)
                            clean_data_x = []
                            clean_data_y = []
                            for i in clean_data:
                                clean_data_x.append(i[0])
                                clean_data_y.append(i[1])
                            try:
                                algorithms_hypervolume[filename.split("_")[4]][filename.split("_")[1]].append(hv.compute(r=ref_point))
                            except:
                                try:
                                    algorithms_hypervolume[filename.split("_")[4]][filename.split("_")[1]] = []
                                    algorithms_hypervolume[filename.split("_")[4]][filename.split("_")[1]] = [hv.compute(r=ref_point)]
                                except:
                                    algorithms_hypervolume[filename.split("_")[4]] = {}
                                    algorithms_hypervolume[filename.split("_")[4]][filename.split("_")[1]] = [hv.compute(r=ref_point)]
                            clean_data = []
                            one_iteration_available = False

    #####################################
    # ploting the hypervolumes -- since most of the data is still not available, this code is just for ehvi with run number zero
    # algorithms_hypervolume["multi_ehvi100gp_dim12_iter100_RUN-6_report.LOG"].reverse()
    # algorithms_hypervolume["multi_nsga2_dim12_iter100_RUN-6_report.LOG"].reverse()
    # plt.plot(range(0, len(algorithms_hypervolume["multi_ehvi100gp_dim12_iter100_RUN-6_report.LOG"])), algorithms_hypervolume["multi_ehvi100gp_dim12_iter100_RUN-6_report.LOG"])
    # plt.plot(100, algorithms_hypervolume["multi_nsga2_dim12_iter100_RUN-6_report.LOG"], 'ro')
    # plt.show()
    for item in algorithms_hypervolume:
        print "Current Item : ", item
        print "Elements of this item: "
        for sub_item in algorithms_hypervolume[item]:
            print sub_item
            print len(algorithms_hypervolume[item][sub_item])
        print "------------------------------------------"
    for run in algorithms_hypervolume:
        plt.figure(figsize=(15.0, 11.0))
        # print run
        legend = []
        max_len = 101
        for algorithm in algorithms_hypervolume[run]:
            legend.append(algorithm)
            if (algorithm != "nsga2"): # I add parego for now, since it's an issue
                algorithms_hypervolume[run][algorithm].reverse()
                plt.plot(range(0, len(algorithms_hypervolume[run][algorithm])),algorithms_hypervolume[run][algorithm],
                         marker='o', linestyle='--')
            else:
                step = int(max_len/len(algorithms_hypervolume[run][algorithm]))
                stop_range = step * len(algorithms_hypervolume[run][algorithm])
                plt.plot(range(0, stop_range, step),algorithms_hypervolume[run][algorithm],
                         marker='o', linestyle='--')
        plt.legend(legend)
        plt.xlim([0, 120])
        figure_name = "Hypervolume_evolution_"+run+".png"
        plt.title(figure_name)
        plt.savefig(figure_name)
        # print "-----------------------------------------"
ref_point = (2, 2) #x is the 1-speed, y is the variance in height
evolution_tracking ("final_experiment_damageWithGoodPriorONE_2D_v2_noIK_ant.ttt", ref_point, 10)