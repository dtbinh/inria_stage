import sys
import os
import math
import numpy as np
import traceback
# Read the parameters coming from the C file
robot_condition = "damage2"  # todo: fix this piece of shit!!! make the input come from outside
problem_dimension = sys.argv[1]

test_id = sys.argv[2]

if len(sys.argv[3:]) > 1:  # it means that the parameters coming are space separated.
    phase_para = sys.argv[3:]
else:  # it means that the parameters coming are comma separated.
    phase_para = sys.argv[3].split(",")
all_para = ",".join(sys.argv[2:])
# print "all_para = ",all_para

def get_readings(meta_data):
    final_hash = {}
    for item in meta_data:
        hash_key, hash_value = item.split("=")
        final_hash[hash_key] = float(hash_value)
    return final_hash


def get_data_variance(height_lines):
    numeric_height_data = []
    for line in height_lines:
        numeric_height_data.append(abs(float(line)))
    variance = np.var(numeric_height_data)
    return variance

###########################
# Do the simulation
###########################
sucecess = False
while sucecess == False:
    try:
        current_path = os.getcwd()
        os.chdir("/home/omohamme/Downloads/V-REP_PRO_EDU_V3_2_0_rev6_64_Linux/")
        # command = "./vrep.sh -h -q -s20000 ./omar_trial/final_experiment_" + robot_condition + "_" + problem_dimension + "D_v2_noIK_ant.ttt -g" + all_para
        # command = "./vrep.sh -h -q -s20000 ./omar_trial/final_experiment_" + robot_condition + "_" + problem_dimension + "D_noIK_ant.ttt -g" + all_para
        # command = "./vrep.sh -h -q -s20000 ./omar_trial/final_experiment_" + robot_condition + "_" + problem_dimension + "D_noIK_ant.ttt -g" + all_para
        command = "./vrep.sh -h -q -s20000 ./omar_trial/final_experiment_" + robot_condition + "_6only_noIK_ant.ttt -g" + all_para
        os.system(command)
        os.chdir(current_path)
        ###############################
        # Read the resulting files
        ###############################
        sim_path = "/home/omohamme/Downloads/V-REP_PRO_EDU_V3_2_0_rev6_64_Linux/"
        command = "rm -rf "

        file_name = sim_path + "positionx_time_" + test_id + ".txt"
        command += file_name + " "
        pos_file = open(file_name, "r")
        pos_data_meta = pos_file.readline().split(",")
        pos_data = get_readings(pos_data_meta)
        pos_file.close()
        os.system("rm -rf " + file_name)

        file_name = sim_path + "height_" + test_id + ".txt"
        command += file_name + " "
        height_file = open(file_name, "r")
        height_data_meta = height_file.readlines()
        height_data = get_data_variance(height_data_meta)
        height_file.close()
        os.system("rm -rf " + file_name)

        # file_name = sim_path + "acc_readings_" + test_id + ".txt" #this is the same parsing as position file
        # command += file_name + " "
        # acc_file = open(file_name, "r")
        # acc_data_meta = acc_file.readlines()
        # acc_data = get_data_variance(acc_data_meta)
        # acc_file.close()
        # os.system("rm -rf " + file_name)

        velocity = pos_data["velocityX"]
        file_name = "output_" + test_id + ".txt"
        output_file = open(file_name, "w")
        print >> output_file, height_data, " ", velocity
        # print >> output_file, acc_data, " ", velocity
        output_file.close()

        sucecess = True
        print "Test is ended for test ID => ", test_id
    except Exception, err:
        sucecess = False
        print err

##############################################
# Process the data to generate useful info
##############################################
# R = math.sqrt((acc_data["accData_x_mean"]**2) + (acc_data["accData_y_mean"]**2) + (acc_data["accData_z_mean"]**2))
# Axr = math.acos(acc_data["accData_x_mean"] / R) * (180 / math.pi)
# Ayr = math.acos(acc_data["accData_y_mean"] / R) * (180 / math.pi)
# Azr = math.acos(acc_data["accData_z_mean"] / R) * (180 / math.pi)

velocity = pos_data["velocityX"]
file_name = "output_" + test_id + ".txt"
output_file = open(file_name, "w")
print >> output_file, height_data, " ", velocity
# print >> output_file, acc_data, " ", velocity
output_file.close()
# print "PYTHON ENDED FOR = ", all_para
