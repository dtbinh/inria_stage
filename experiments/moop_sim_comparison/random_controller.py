__author__ = 'omohamme'
# The target from this file is to make a function that generates random solutions. These solutions will be used
# as a baseline comparison.
import random
import os
def random_number_gen(start_num, stop_num, dimension):
    results = []
    for i in range(dimension):
        random_number = start_num + (random.random() * (stop_num - start_num))
        results.append(str(random_number))
    return results

def random_robot_controller(start_test, end_test, number_of_generations, dimension, scene_name):
    actual_dimension = int(dimension[3:])
    actual_generations = int(number_of_generations[4:])
    start = 0.0
    stop = 3.14
    for iteration in range(start_test, end_test+1, 1):
        final_points = ["#######################The current pareto points"]
        for generation in range(actual_generations):
            generated_parameters = random_number_gen(start, stop, actual_dimension)
            test_id = "pythonRandom"
            generated_parameters.insert(0, test_id)
            command = "python /home/omohamme/INRIA/experiments/moop_sim_comparison/test_case_service.py " + " ".join(generated_parameters)
            os.system(command)
            output_file = "output_"+test_id+".txt"
            with open(output_file, "r") as file:
                # results_data = ",".join(file.readlines()[0].split("   ")) + ","
                results_data = file.readlines()[0].split("   ")
                results_data[0] = str(-1.0 * float(results_data[0]))
                results_data[1] = str(-1.0 * (1.0 - float(results_data[1])))
                results_data = ",".join(results_data)
                final_points.append(results_data)
        final_points.append("##############################################")
        # multi_parego_dim12_iter100_RUN-1_report.LOG
        final_file = "multi_random_"+dimension.lower()+"_" + number_of_generations.lower() + "_RUN-"+str(iteration) + "_report.LOG"
        # final_file = "./log_files/" + scene_name[:-4] + "/"+ final_file
        final_file = "./log_files/" + scene_name[:-4] + "/" + final_file
        with open(final_file, "w") as file:
            print >> file, "\n".join(final_points)

# random_robot_controller(10, "ITER100", "DIM12", "final_experiment_good_7_noIK_ant.ttt")
start_test = 10
end_test = 19
random_robot_controller(start_test, end_test, "ITER100", "DIM12", "final_experiment_good_7_noIK_ant.ttt")