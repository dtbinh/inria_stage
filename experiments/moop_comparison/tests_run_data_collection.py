#This script will run all the tests, and record their results in different files

#I need to add methods for:
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
import sys
import time

current_run = 0

def delete_old_log_files():
    log_file_name = "*_report.LOG"
    log_file_path = "/home/omohamme/INRIA/experiments/moop_comparison/log_files/" + log_file_name
    command = "rm -rf "+log_file_path
    os.system(command)
    print "All prev log files have been deleted "
    log_file_name = "*.png"
    log_file_path = "/home/omohamme/INRIA/experiments/moop_comparison/figures/" + log_file_name
    command = "rm -rf "+log_file_path
    os.system(command)
    log_file_path = "/home/omohamme/INRIA/experiments/moop_comparison/boxplots/" + log_file_name
    command = "rm -rf "+log_file_path
    os.system(command)    
    print "All prev figure files have been deleted "    
    
def test_run (testcase,new_log_files = False):
    global current_run
    log_file_name = testcase + "_RUN-" + str(current_run) +"_report.LOG"
    log_file_path = "/home/omohamme/INRIA/experiments/moop_comparison/log_files/" + log_file_name
    command = "/home/omohamme/INRIA/Software/folders/limbo_master/limbo/build/src/benchmarks/"+testcase+" > " + log_file_path
    os.system(command)
    print "Testcase Ended : ",(testcase + "_RUN-" + str(current_run))

def test_run_main(testcases_name, num_of_runs = 1, parallel = True):
    global current_run
    #main body
    current_run = 0
    t0 = time.time()
    for counter in range(num_of_runs):
        current_run = counter
        if parallel == True:
            pool = multiprocessing.Pool(processes=6)
            pool.map(test_run,testcases_name)
        else:
            map(test_run,testcases_name)
    t1 = time.time() - t0
    print "Total regression time = ",t1
