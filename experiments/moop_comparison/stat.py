# This is the statistical module.
import numpy as np;
import matplotlib.pyplot as plt;
import os
import sys
from scipy import stats
import pandas

algorithms = ["PAREGO", "NSGA2", "EHVI", "NSGA25p4g","NSGA100p1g"]
dimensions = ["DIM2", "DIM6"]
problems = ['ZDT1', 'ZDT2']
iterations = ["ITER100"]

hyper_vol_file = open("hyper_vol_report.txt", "r")
hyper_vol_lines = hyper_vol_file.read().splitlines()
hyper_vol_file.close()

final_data_frame = {}  # Will be converted to a data frame at the end.
iteration = []
algorithm = []
problem = []
dim = []
hypervolume = []
run = []

stat_file = open("Stat_tests.txt", "w")

for line in hyper_vol_lines:
    name, hyp_vol = line.split(":::")
    hyp_vol = float(hyp_vol)
    name = name.split("_")[1:]  # ignore the first word, multi
    algorithm.append(name[0])
    problem.append(name[1])
    dim.append(name[2])
    iteration.append(name[3])
    run.append(name[4])
    hypervolume.append(hyp_vol)

final_data_frame = {"algorithm": algorithm, "problem": problem, "dim": dim, "run": run, "iteration": iteration,
                    "hypervolume": hypervolume}
final_data_frame = pandas.DataFrame(final_data_frame)
final_data_frame = final_data_frame.sort(["iteration", "run", "dim", "problem"])
print final_data_frame

# draw all the possible boxplots
print list(final_data_frame["algorithm"])
print "----------------"
print algorithms
print >> stat_file, "This is a comparison between different algorithms. Always the base case is the PAREGO results\n\n"
for problem in problems:
    for dimen in dimensions:
        data = []
        labels = []
        for algo in algorithms:
            data.append(final_data_frame[(final_data_frame["algorithm"] == algo.lower()) & (
                final_data_frame["dim"] == dimen.lower()) &
                                         (
                                             final_data_frame["problem"] == problem.lower())].hypervolume.values.T.tolist())
            labels.append(algo)

        # This part is for the statisitcs
        # Here, I will always use my base case as the Parego algorithm - Since, from the boxplots - it performed the worst by far.
        print >> stat_file, "Current problem = ", problem, " -- Current dimension = ", dimen
        for data_index in range(len(data)):  # Take PAREGO as your base case
            data_item = data[data_index]
            if data_index != 0:
                try:
                    x = stats.wilcoxon(data[0], data_item)
                    print >> stat_file, "PAREGO VS ", labels[data_index], " -->", x
                    print >> stat_file, "PAREGO median = ", np.median(data[0])
                    print >> stat_file, labels[data_index], " median = ", np.median(data_item)
                except:
                    # print >> sys.stderr, 'There is a data matching case here. Take care!\n - data:', data, "\n problem = ", problem, "\n dimen = ", dimen, "\n labels = ", labels
                    pass
        for data_index in range(len(data)):  # Take EHVI as your base case
            data_item = data[data_index]
            if (data_index != 2) and (data_index != 0):
                try:
                    x = stats.wilcoxon(data[2], data_item)
                    print >> stat_file, "EHVI VS ", labels[data_index], " -->", x
                    print >> stat_file, "EHVI median = ", np.median(data[2])
                    print >> stat_file, labels[data_index], " median = ", np.median(data_item)
                except:
                    # print >> sys.stderr, 'There is a data matching case here. Take care!\n - data:', data, "\n problem = ", problem, "\n dimen = ", dimen, "\n labels = ", labels
                    pass
        print >> stat_file, "----------------------------------------------------------"
        # This part is for drawing the different boxplots
        figure_name = problem + "_" + dimen + "_.png"
        current_path = os.getcwd()
        os.chdir("/home/omohamme/INRIA/experiments/moop_comparison/boxplots/")
        # plt.figure(figsize=(15.0, 11.0))
        plt.figure()
        plt.boxplot(data)
        plt.xticks(range(1, len(data) + 1), labels, fontsize = 6, rotation = 45)
        plt.yticks(fontsize = 15)
        plt.ylabel("The Hypervolume size")
        figure_title = " ".join((figure_name.split(".")[0]).split("_"))
        print figure_title
        plt.title(figure_title)
        print figure_name
        plt.savefig(figure_name)
        os.chdir(current_path)

stat_file.close()
