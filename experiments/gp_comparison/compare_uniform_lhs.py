__author__ = 'omohamme'
# I want here to find a good model
from utilities import *

# os.system('cls' if os.name == 'nt' else 'clear')
os.system("reset")
objectives_list = [1]
# toolbar_method = toolbar(toolbar_width=10)
# toolbar_method.init_toolbar()
num_sampling_points = 200
num_test_points = 1000
for dimension in [6]:
    print "########################################################"
    print "--- Testing Dimension : ", dimension
    print "########################################################"
    models_scores_good_robot = dict(robot_speed_uniform=[],
                        robot_speed_lhs=[])
    for i in range(40): #Repeat this exp 100 times
        print "Current iterations = ", i
        learninig_model = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget=3.00e-13)
        robot_speed_uniform = robot_learning(num_in_samples=num_sampling_points, objectives_list=objectives_list, learninig_model=copy.deepcopy(learninig_model), robot_condition="good", dimension=dimension, sampling_function=data_parser)
        robot_speed_lhs = robot_learning(num_in_samples=num_sampling_points, objectives_list=objectives_list, learninig_model=copy.deepcopy(learninig_model), robot_condition="good", dimension=dimension, sampling_function=lhs_sampling_robot)

        # Initializing and fitting the models.
        robot_speed_uniform.get_training_data()
        robot_speed_uniform.fit_learning_model()

        robot_speed_lhs.get_training_data()
        robot_speed_lhs.fit_learning_model()

        test_method = robot_test_samples(num_test_samples=num_test_points, objectives_list=objectives_list, robot_condition="good", dimension=dimension)
        test_method.get_test_samples()
        test_points, test_results = test_method.return_test_pairs()

        models_scores_good_robot["robot_speed_uniform"].append(repr(r2_score(test_results[0], robot_speed_uniform.predict(test_points))))
        models_scores_good_robot["robot_speed_lhs"].append(repr(r2_score(test_results[0], robot_speed_lhs.predict(test_points))))

        for item in models_scores_good_robot:
            print "item : ", item, " --> ", models_scores_good_robot[item]

        # Delete all the objects and variables you've since this eats a huge amount of memory after.
        gc.collect()
        print "--------------------------------------------------------------------"

    combined_results = []
    labels = []
    for item in models_scores_good_robot:
        combined_results.append(map(float, models_scores_good_robot[item]))
        labels.append(item)

    current_path = os.getcwd()
    plt.figure(figsize=(15.0, 11.0))
    plt.boxplot(combined_results)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=20, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("R2 Score", fontsize=10)
    plt.xlabel("Model to be tested", fontsize=10)
    plt.savefig("compare_samplingTechniques_" + str(objectives_list[0]) + "objective_" + str(dimension) + "D_" + str(num_sampling_points) + "points.png")
    os.chdir(current_path)

    del models_scores_good_robot
# Conclusions from this really nice
# DecisionTreeRegressor(max_depth=20) with 70000 give a very good result for the 12-D controller!
# I always thought decision trees are really stupid, but it fucken works!!!
# Out of caution, lets generate more data.
# Also, in the optimization problem, let's repeat the test for 20 times, and take the model with the highest score.
