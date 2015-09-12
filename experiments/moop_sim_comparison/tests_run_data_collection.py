# This script will run all the tests, and record their results in different files

# I need to add methods for:
## - A timer to stop the test after certain time. This is important for the grid work.
## - A Method to send an email with the current run report to my personal email.
## - Management system: This is critical, especially when I've to divide the test between different machines. There should be an online
## document - for example - that contain information about what tests are done, what test to work on, and so on.
## - A method to clean the system from my processes as soon as the time is over. This is for the grid work mainly, since I should not
## leave any process by accident after I finish my time. Probably, it should kill all the process with my name in. I still have to check.
## This is going to be a master piece of Engineering :D

# General issues: 
# - Should I categorized each group of testcases in their own folders? For example, all iteration 30 will go into one folder, and 
# inside this folder, there will be separate folders for each problem? --> Too complicated.
import multiprocessing
import os
import time

current_run = 0
current_path = os.getcwd()
current_scene = ""


def delete_old_log_files():
    log_file_path = current_path + "/log_files/*"
    command = "rm -rf " + log_file_path
    os.system(command)
    print "All prev log files have been deleted "
    log_file_path = current_path + "/figures/*"
    command = "rm -rf " + log_file_path
    os.system(command)
    log_file_path = current_path + "/boxplots/*"
    command = "rm -rf " + log_file_path
    os.system(command)
    print "All prev figure files have been deleted "


def create_folder_heirarichy(experiments_names):
    # Create the folder heirarichy
    if not os.path.isdir(current_path + "/log_files"):
        os.makedirs(current_path + "/log_files")
    if not os.path.isdir(current_path + "/boxplots"):
        os.makedirs(current_path + "/boxplots")
    if not os.path.isdir(current_path + "/figures"):
        os.makedirs(current_path + "/figures")
    if not os.path.isdir(current_path + "/pareto_comparison"):
        os.makedirs(current_path + "/pareto_comparison")

    for experiment in experiments_names:
        mypath = current_path + "/log_files/" + experiment[:-4]
        if not os.path.isdir(mypath):
            os.makedirs(mypath)

        mypath = current_path + "/boxplots/" + experiment[:-4]
        if not os.path.isdir(mypath):
            os.makedirs(mypath)

        mypath = current_path + "/figures/" + experiment[:-4]
        if not os.path.isdir(mypath):
            os.makedirs(mypath)

        mypath = current_path + "/pareto_comparison/" + experiment[:-4]
        if not os.path.isdir(mypath):
            os.makedirs(mypath)


def test_run(testcase):
    global current_scene
    testcase_name = testcase[0]
    log_file_path = current_path + "/log_files/" + current_scene + "/" + testcase[1]
    command = "/home/omohamme/INRIA/Software/folders/limbo_master/limbo/build/src/benchmarks/" + testcase_name + " > " + log_file_path
    deadline = time.time() + testcase[2]
    while(time.time() < deadline):
        pass
    print "Testcase started : ", (testcase[1])
    os.system(command)
    print "Testcase Ended : ", (testcase[1])


# def test_run_main(testcases_name, scenes, num_of_runs=1, parallel=True):
def test_run_main(testcases_name, scenes, start_test_index=0, end_test_index=1, parallel=True):
    # main body
    global current_scene
    testcases_logfiles_names = []
    counter = 0
    for testcase in testcases_name:
        for run_iter in range(start_test_index, end_test_index + 1, 1):
            testcases_logfiles_names.append([testcase, testcase + "_RUN-" + str(run_iter) + "_report.LOG", counter])
            counter += 1

    t0 = time.time()
    for each_scene in scenes:
        current_scene = each_scene[:-4]
        if parallel:
            pool = multiprocessing.Pool(processes=3)
            pool.map(test_run, testcases_logfiles_names)
        else:
            map(test_run, testcases_logfiles_names)
    t1 = time.time() - t0
    print "Total regression time = ", t1
