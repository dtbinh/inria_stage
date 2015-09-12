import reports_parsing
import tests_run_data_collection
import generate_test_combinations
import numpy as np;
import matplotlib.pyplot as plt;
from PyGMO import *
import os,sys
#Decide which experiments to run
#dimensions = ["DIM2", "DIM6","DIM15","DIM20","DIM30"]
#iterations = ["ITER30","ITER200"]

#algorithms = ["PAREGO","NSGA2","EHVI"]
#dimensions = ["DIM2", "DIM6","DIM15","DIM20","DIM30"]
#problems = ['ZDT1', 'ZDT2', 'ZDT3', 'MOP2']
#iterations = ["ITER30"]

#algorithms = ["NSGA2","EHVI"]
#dimensions = ["DIM2"]
#problems = ['ZDT2','ZDT3', 'MOP2']
#iterations = ["ITER30"]

#algorithms = ["PAREGO","NSGA2","EHVI"]
#dimensions = ["DIM2", "DIM6"]
#problems = ['ZDT1', 'ZDT2']
#iterations = ["ITER100"]

# algorithms = ["EHVINC2","EHVIN","EHVIC2"]
# dimensions = ["DIM2", "DIM6"]
# problems = ['ZDT1', 'ZDT2']
# iterations = ["ITER100"]
#
# num_of_runs = 10

#generate_test_combinations.gen_config (algorithms,dimensions,problems,iterations)
#
# testcases_names = []
# for alg in algorithms:
#     for prob in problems:
#         for dim in dimensions:
#             for iter in iterations:
#                 testcases_names.append("multi_"+"_".join(map(lambda y:str(y).lower(),(alg,prob,dim,iter))))

#tests_run_data_collection.delete_old_log_files()
#tests_run_data_collection.test_run_main(testcases_name = testcases_names,num_of_runs = num_of_runs)

import glob
log_files_list = glob.glob("/home/omohamme/INRIA/experiments/moop_comparison/log_files/*.LOG")

hypervolume_report = open("hyper_vol_report.txt", "w")
for log_file in log_files_list:
    results = reports_parsing.clean_data_fn(log_file)
    figure_name = results[3]+".png"
    current_path = os.getcwd()
    os.chdir("/home/omohamme/INRIA/experiments/moop_comparison/figures/")    
    plt.figure()
    plt.plot(results[1],results[2],'bo')
    plt.title(results[3])
    plt.savefig(figure_name)
    os.chdir(current_path)

    hv = util.hypervolume(results[0])
    ref_point = (11, 11)
    x = str(log_file.split("/")[-1]) + ":::" + str(hv.compute(r=ref_point)) + "\n"
    hypervolume_report.write(x)
    
hypervolume_report.close()