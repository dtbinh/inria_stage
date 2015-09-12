import reports_parsing
import tests_run_data_collection
import generate_test_combinations
import matplotlib.pyplot as plt
from PyGMO import *
import os
import sys
import glob
# import stats_function
import stats_function_2

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

current_path = os.getcwd()
# Read the parameters coming from the C file

# algorithms = ["EHVI", "NSGA2", "PAREGO"]
# algorithms = ["NSGA2"]
# algorithms = ["EHVISTSA", "EHVILHS", "EHVI100SAMPLE", "EHVI15"]
# algorithms = ["EHVIMATRERN"]
# algorithms = ["BOPT"]
# algorithms = ["EHVI", "EHVI100GP", "NSGA2", "PAREGO"]
# algorithms = ["EHVI100GP"]
# algorithms = ["EHVI11"]
# algorithms = ["EHVI10CMAES"]
# algorithms = ["EHVI", "PAREGO"]
# algorithms = ["EHVIGP2D", "PAREGO", "NSGA2"]
algorithms = ["NSGA2"]
# algorithms = ["EHVI15"]
# algorithms = ["EHVI10NSGAII"]
# algorithms = ["EHVISTSA"]
# algorithms = ["NSGA200P2000G"] # This is the optimal pareto front
# algorithms = ["NSGA200P500G"] # This is the optimal pareto front
# algorithms = ["NSGA30p50g"]
# algorithms = ["PAREGO", "NSGA2"]
# dimensions = ["DIM12"]
dimensions = ["DIM2"]
iterations = ["ITER100"]
# scenes = ["final_experiment_good_6.ttt", "final_experiment_good_7.ttt", "final_experiment_damaged_6.ttt",
#           "final_experiment_damaged_7.ttt", "final_experiment_good_12.ttt", "final_experiment_damaged_12.ttt"]
# scenes = ["final_experiment_good_6.ttt"]
# scenes = ["final_experiment_good_7.ttt"]
# scenes = ["final_experiment_good_7_ayr.ttt"]
# scenes = ["all_minimization_trial.ttt"]
# scenes = ["final_experiment_good_7_noIK_ant_optimalParetoFront.ttt"]
# scenes = ["final_experiment_good_7_noIK_ant_single_BOPT.ttt"]
# scenes = ["final_experiment_damaged1_7_noIK_ant.ttt"]
# scenes = ["final_experiment_damaged2_7_noIK_ant.ttt"]
# scenes = ["final_experiment_good_7_noIK_ant_clean.ttt"]
# scenes = ["final_experiment_damageW`ithGoodPriorTWO_2D_v2_noIK_ant_NEW.ttt"]
# scenes = ["final_experiment_damageWithGoodPriorTWO_2D_v2_noIK_ant.ttt"]
# scenes = ["final_experiment_damageWithGoodPriorONE_2D_v2_noIK_ant.ttt"]
scenes = ["final_experiment_good_12D_noIK_ant.ttt"]
start_test = 0
end_test = 19
num_of_runs = end_test + 1

# generate_test_combinations.gen_config(algorithms, dimensions, iterations)

testcases_names = []
for alg in algorithms:
    for dim in dimensions:
        for current_iteration in iterations:
            testcases_names.append("multi_" + "_".join(map(lambda y: str(y).lower(), (alg, dim, current_iteration))))

tests_run_data_collection.create_folder_heirarichy(scenes)
# tests_run_data_collection.test_run_main(testcases_name=testcases_names, scenes=scenes, start_test_index=start_test,
#                                         end_test_index=end_test, parallel=True)
#
for scene in scenes:
    log_files_list = glob.glob(current_path + "/log_files/" + scene[:-4] + "/*.LOG")
    pareto_front_comparison = {}
    hypervolume_report = open("hyper_vol_report_" + scene[0:-4] + ".txt", "w")
    for log_file in log_files_list:
        results = reports_parsing.clean_data_fn(log_file)
        clean_x, clean_y = pareto_frontier(results[1], results[2])
        figure_name = results[3] + ".png"
        current_path = os.getcwd()
        os.chdir(current_path + "/figures/" + scene[:-4])
        plt.figure()
        plt.plot(clean_x, clean_y, 'bo')
        plt.title(results[3])
        plt.savefig(figure_name)
        os.chdir(current_path)
        clean_data = []
        for i in range(len(clean_x)):
            clean_data.append((clean_x[i], clean_y[i]))
        hv = util.hypervolume(clean_data)
        ref_point = (2, 2) # x is the 1-speed, y is the variance in height
        x = str(log_file.split("/")[-1]) + ":::" + str(hv.compute(r=ref_point)) + "\n"
        hypervolume_report.write(x)

        try:
            pareto_front_comparison[log_file.split("_")[4]][log_file.split("_")[1]] = [clean_x, clean_y]
        except:
            try:
                pareto_front_comparison[results[3].split("_")[4]][results[3].split("_")[1]] = []
                pareto_front_comparison[results[3].split("_")[4]][results[3].split("_")[1]] = [clean_x, clean_y]
            except:
                pareto_front_comparison[results[3].split("_")[4]] = {}
                pareto_front_comparison[results[3].split("_")[4]][results[3].split("_")[1]] = [clean_x, clean_y]

    hypervolume_report.close()

    # Perform the stats on this hypervolume report.
    stats_function_2.stats_fn(scene, num_of_runs)
    for current_run in pareto_front_comparison:
        current_path = os.getcwd()
        os.chdir(current_path + "/pareto_comparison/" + scene[:-4])
        plt.figure()
        legend = []
        figure_name = "pareto_comparison_"+current_run
        for algorithm in pareto_front_comparison[current_run]:
            plt.plot(pareto_front_comparison[current_run][algorithm][0], pareto_front_comparison[current_run][algorithm][1], marker='o', linestyle='--')
            legend.append(algorithm)
        plt.title(figure_name)
        plt.legend(legend)
        plt.savefig(figure_name)
        os.chdir(current_path)