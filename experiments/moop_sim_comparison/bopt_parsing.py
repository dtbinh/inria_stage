# Reports parsing facility
# We have two types of reports to parse, and we want to map them to the same data structure
# The first type is the of Parego, EHVI, NSBO. The 2nd type is that of NSGA2
# To get the paramters, I need to first parse the name of the file. This ontains all the information I need.

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy import stats


scene = "final_experiment_good_7_noIK_ant_single_BOPT.ttt"
num_iterations = 10

def clean_data_fn(log_file):
    with open(log_file, "r") as file:
        log_data = file.read().splitlines()

    # The important line for me is the one before the end log_data[:-2]
    if "random" not in log_file:
        line = log_data[-2].replace(" ", "")
        clean_data = float(line.split("=")[1])
    else:
        max_speed = 100
        # print log_data
        for line in log_data[1:-1]:
            line_split = line.split(",")
            current_speed = float(line_split[1])
            if current_speed < max_speed:
                max_speed = current_speed
        clean_data = max_speed
    print "clean_data = ", clean_data
    return clean_data

# Stats function
def stats_fn(data_frame):
    global scene
    stat_file = open("Stat_tests_" + scene[:-4] + ".txt", "w")
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
    plt.xticks(range(1, len(data_frame.keys()) + 1), data_frame.keys())
    plt.title(figure_name)
    plt.savefig(figure_name)
    os.chdir(current_path)

    stat_file.close()


current_path = os.getcwd()
log_files_list = glob.glob(current_path + "/log_files/" + scene[:-4] + "/*.LOG")
data_frame = {}
print len(log_files_list)
for log_file in log_files_list:
    data = clean_data_fn(log_file=log_file)
    file_name = log_file.split("/")[-1]
    algorithm = file_name.split("_")[1]
    run_number = int(file_name.split("_")[4].split("-")[1])
    try:
        data_frame[algorithm].append(data)
    except:
        data_frame[algorithm] = [data]

for item in data_frame:
    print item
    print data_frame[item]
    print len(data_frame[item])
    print "-----------------------------------"
stats_fn(data_frame)