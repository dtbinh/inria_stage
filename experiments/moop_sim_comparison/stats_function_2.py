__author__ = 'omohamme'
# This is the statistical module.
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def stats_fn(scene, num_iterations):
    hyper_vol_file = open("hyper_vol_report_"+scene[0:-4]+".txt", "r")
    hyper_vol_lines = hyper_vol_file.read().splitlines()
    hyper_vol_file.close()

    data_frame = {}
    stat_file = open("Stat_tests_" + scene[:-4] + ".txt", "w")

    for line in hyper_vol_lines:
        print line
        name, hypervolume = line.split(":::")
        hypervolume = float(hypervolume)
        name_list = name.split("_")[1:]  # ignore the first word, multi
        run_number = int(name.split("_")[4].split("-")[1])
        algorithm = name_list[0]
        try:
            data_frame[algorithm][run_number] = hypervolume
        except:
            x = []
            for i in range(num_iterations):
                x.append(0)
            data_frame[algorithm] = x[:]
            data_frame[algorithm][run_number] = hypervolume
    print data_frame
    # This part is for the statistics
    # Here, I will always use my base case as the PAREGO algorithm - Since, from the boxplots - it performed
    # the worst by far.
    seen_pairs = []
    for algorithm in data_frame:
        for algorithm2 in data_frame:
            if (algorithm != algorithm2) and ((algorithm, algorithm2) not in seen_pairs):
                seen_pairs.append((algorithm, algorithm2))
                seen_pairs.append((algorithm2, algorithm))
                statistical_significance = stats.wilcoxon(data_frame[algorithm], data_frame[algorithm2])
                print >> stat_file, algorithm, " VS ", algorithm2, " -->", statistical_significance
                print >> stat_file, algorithm, " median = ", np.median(data_frame[algorithm])
                print >> stat_file, algorithm2, " median = ", np.median(data_frame[algorithm2])
                print >> stat_file, "----------------------------------------------------------"

    # # This part is for drawing the different boxplots
    figure_name = scene + "_.png"
    current_path = os.getcwd()
    os.chdir("/home/omohamme/INRIA/experiments/moop_sim_comparison/boxplots/" + scene[:-4] + "/")
    plt.figure(figsize=(15.0, 11.0))
    plt.boxplot(data_frame.values())
    plt.xticks(range(1, len(data_frame.keys()) + 1), data_frame.keys(), rotation=45, fontsize=20)
    plt.yticks(fontsize=15)
    # plt.title(" ".join(scene[:-4].split("_")))
    plt.ylabel("Hypervolume area", fontsize=20)
    plt.savefig(figure_name)
    os.chdir(current_path)

    stat_file.close()

