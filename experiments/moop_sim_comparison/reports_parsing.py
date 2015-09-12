# Reports parsing facility
# We have two types of reports to parse, and we want to map them to the same data structure
# The first type is the of Parego, EHVI, NSBO. The 2nd type is that of NSGA2
# To get the paramters, I need to first parse the name of the file. This ontains all the information I need.

import glob
import numpy as np
log_files_list = glob.glob(
    "/home/omohamme/INRIA/experiments/moop_sim_comparison/log_files/*.LOG")  # This address will be changed later to
    # the big directoy, where all log files are + .log will be .LOG .

def is_nsga2(file_name):
    for parts in file_name:
        if "nsga" in parts:
            return True
    return False


# print log_files_list
def clean_data_fn(log_file):
    clean_data = []
    coordinates = []
    test_name = log_file.split("/")[-1]
    with open(log_file, "r") as file:
        log_data = file.read().splitlines()

    file_type = test_name.split("_")

    # Parse based on the data type
    # if "nsga" not in file_type:
    log_data.reverse()
    read_flag = False
    for line in log_data:
        if line == "##############################################":
            read_flag = True
        elif line == "#######################The current pareto points":
            read_flag = False
            # I should make a fucken break HERE!!!
            break
        else:
            if read_flag == True:
                if "random" not in log_file:
                    data = line.split("---->")#[-1].split(",")[:-1]
                    # This piece of code is really overfitting the reports. I should take care of it soon.
                    if len(data[0].split(",")[:-1]) == 2: # That means that this part has the pareto points.
                        good_point = data[0].split(",")[:-1]
                        points = data[1].split(",")[:-1]#.split(",")[:-1]
                    else:
                        good_point = data[1].split(",")[:-1]
                        points = data[0].split(",")[:-1]#.split(",")[:-1]
                    ######################################################################################
                    data_numeric = []
                    # for i in data:  # convert data from string format to float format
                    # data_numeric.append(float(data[0]))  # I added the minus here. Result of my work with Mathew
                    # data_numeric.append(-float(data[1]))  # I added the minus here. Result of my work with Mathew
                    # clean_data.append(data_numeric)

                    data_numeric.append(-float(good_point[0]))  # I added the minus here. Result of my work with Mathew
                    data_numeric.append(-float(good_point[1]))  # I added the minus here. Result of my work with Mathew
                    clean_points = []
                    for i in points:
                        clean_points.append(float(i))
                    coordinates.append(clean_points)
                    clean_data.append(data_numeric)
                else:
                    data = line.split(",")
                    data_numeric = []
                    data_numeric.append(-float(data[1]))  # I added the minus here. Result of my work with Mathew
                    data_numeric.append(-float(data[0]))  # I added the minus here. Result of my work with Mathew
                    clean_data.append(data_numeric)

    clean_data_x = []
    clean_data_y = []

    new_file = open(log_file+"_CLEAN", "w") #write the clean data with the pareo front in a new file
    counter = 0
    for i in clean_data:
        clean_data_x.append(i[0])
        clean_data_y.append(i[1])
        if "random" not in log_file:
            coord = 3.14 * np.array(coordinates[counter])
            pareto = np.array([1, 0]) - np.array(i)
            # pareto = 1 - i[0]
            # print >> new_file, i, "---->", coordinates[counter]
            print >> new_file, list(pareto), "---->", list(coord)
            counter += 1
    new_file.close()

    return clean_data, clean_data_x, clean_data_y, test_name
