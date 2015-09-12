__author__ = 'omohamme'

import random
# file_name = "all_outputs_" + str(controller_dim) + "d_" + robot_condition + ".txt"
file_name = "all_outputs_2d_V2Controller_good.txt"
samples = []
observations = []
for i in range(50):
    num_samples = 200
    try:
        with open(file_name, "r") as file:
            file_lines = file.read().splitlines()[200:]
            random.shuffle(file_lines) # This new line targets to shuffle the data.
            # This should make the sample more representative and fair.
    except ValueError:
        print "There is no such data file ..... "
        exit()

    # Open a file to write in
    new_file_name = "all_outputs_2d_V2Controller_good_" + str(i) + ".txt"
    with open(new_file_name, "w") as file:
        random_indices = random.sample(range(0, len(file_lines)), num_samples)
        for i in range(num_samples):
            # Get a random index
            random_index = random_indices[i]
            # Get the values
            line = file_lines[random_index]
            input_points = line.split("--->")[0].split(",")
            input_points = map(float, input_points)
            input_points = [x/3.14 for x in input_points]
            output_points = line.split("--->")[1].replace(" ", "").split(",")
            # output_points = map(float, output_points)[target] # Get the target we want.
            output_points = map(float, output_points)# Get the target we want and deduce the mean from it
            output_points.reverse()
            x = map(str, input_points + output_points)
            print >> file, " ".join(x)