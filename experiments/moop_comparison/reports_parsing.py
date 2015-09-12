#Reports parsing facility
#We have two types of reports to parse, and we want to map them to the same data structure
#The first type is the of Parego, EHVI, NSBO. The 2nd type is that of NSGA2
#To get the paramters, I need to first parse the name of the file. This ontains all the information I need.

import glob
log_files_list = glob.glob("/home/omohamme/INRIA/experiments/moop_comparison/log_files/*.LOG") #This address will be changed later to the big directoy, where all log files are + .log will be .LOG .
def is_nsga2 (file_name):
    for parts in file_name:
        if "nsga" in parts:
            return True
    return False
#print log_files_list
def clean_data_fn (log_file):
    clean_data = []
    test_name = log_file.split("/")[-1]
    with open(log_file,"r") as file:
        log_data = file.read().splitlines()
    
    file_type = test_name.split("_")
    
    #Parse based on the data type
    #if "nsga" not in file_type:
    if not is_nsga2(file_type):
        log_data.reverse()
        read_flag = False
        for line in log_data:
            if line == "##############################################":
                read_flag = True
            elif line == "#######################The current pareto points":
                read_flag = False
            else:
                if read_flag == True:
                    data = line.split(",")[:-1]
                    data_numeric = []
                    for i in data: #convert data from string format to float format
                        data_numeric.append(-float(i)) #I added the minus here. Result of my work with Mathew
                    clean_data.append(data_numeric)
    else:
        #log_data = log_data #Last line is the timing
        read_flag = False
        #for line in log_data[:-1]:
        for line in log_data:
            data = line.split(",")[:-1]
            data_numeric = []
            for i in data: #convert data from string format to float format
                data_numeric.append(-float(i))#I added the minus here. Result of my work with Mathew
            clean_data.append(data_numeric)
    
    clean_data_x = []
    clean_data_y = []
    
    for i in clean_data:
        clean_data_x.append(i[0])
        clean_data_y.append(i[1])
        
    return clean_data,clean_data_x,clean_data_y,test_name