__author__ = 'omohamme'
# I want here to find a good model
from utilities import *

# os.system('cls' if os.name == 'nt' else 'clear')
os.system("reset")
objectives_list = [0]
# toolbar_method = toolbar(toolbar_width=10)
# toolbar_method.init_toolbar()
models_scores_good_robot_0 = []
models_scores_good_robot_1 = []
models_scores_good_robot_2 = []
num_sampling_points = 200
number_of_estimators = 1
num_test_points = 1000
for dimension in [12]:
    print "########################################################"
    print "--- Testing Dimension : ", dimension
    print "########################################################"
    models_scores_good_robot = dict(robot_multiple_regressors_model=[],
                        robot_multiple_regressors_model_cluster=[],
                        robot_learning_gp=[],
                        robot_fit_cluster_centroids=[],
                         )
    for i in range(20): #Repeat this exp 100 times
        print "Current iterations = ", i
        learninig_model = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget=3.00e-13)
        # robot_multiple_regressors_model = robot_multiple_regressors(number_of_estimators=number_of_estimators, learninig_model=copy.deepcopy(learninig_model), num_in_samples=num_sampling_points, dimension = dimension, objectives_list=objectives_list)
        # robot_multiple_regressors_model_cluster = robot_multiple_regressors_clustering(number_of_estimators=number_of_estimators, learninig_model=copy.deepcopy(learninig_model), num_in_samples=num_sampling_points, dimension = dimension, objectives_list=objectives_list)
        robot_learning_gp = robot_learning(num_in_samples=num_sampling_points*number_of_estimators, objectives_list=objectives_list, learninig_model=copy.deepcopy(learninig_model), robot_condition="good", dimension=dimension)
        robot_fit_cluster_centroids = robot_multiple_regressors_ClusterCentroids(num_in_samples=num_sampling_points*number_of_estimators, robot_condition="good", dimension=dimension, learninig_model=copy.deepcopy(learninig_model), objectives_list=objectives_list)

        # Initializing and fitting the models.
        # robot_multiple_regressors_model.initialization()
        # robot_multiple_regressors_model.fit()
        # robot_multiple_regressors_model_cluster.initialization()
        # robot_multiple_regressors_model_cluster.fit()
        robot_fit_cluster_centroids.initialization()
        robot_fit_cluster_centroids.fit()
        robot_learning_gp.get_training_data()
        robot_learning_gp.fit_learning_model()

        test_method = robot_test_samples(num_test_samples=num_test_points, objectives_list=objectives_list, robot_condition="good", dimension=dimension)
        test_method.get_test_samples()
        test_points, test_results = test_method.return_test_pairs()

        # models_scores_good_robot["robot_multiple_regressors_model"].append(repr(r2_score(test_results[0], robot_multiple_regressors_model.predict(test_points))))
        # models_scores_good_robot["robot_multiple_regressors_model_cluster"].append(repr(r2_score(test_results[0], robot_multiple_regressors_model_cluster.predict(test_points))))
        models_scores_good_robot["robot_learning_gp"].append(repr(r2_score(test_results[0], robot_learning_gp.predict(test_points))))
        models_scores_good_robot["robot_fit_cluster_centroids"].append(repr(r2_score(test_results[0], robot_fit_cluster_centroids.predict(test_points))))

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
    # plt.savefig("find_good_reg_model_" + str(objectives_list[0]) + "objective_" + str(dimension) + "D_" + str(number_of_estimators) + "GP_" + str(num_sampling_points) + "points_V2Controller.png")
    plt.savefig("robot_controller_v4_" + str(objectives_list[0]) + "objective_" + str(dimension) + "D_" + str(number_of_estimators) + "GP_" + str(num_sampling_points) + "points_V2Controller.png")
    os.chdir(current_path)

    del models_scores_good_robot
# Conclusions from this really nice
# DecisionTreeRegressor(max_depth=20) with 70000 give a very good result for the 12-D controller!
# I always thought decision trees are really stupid, but it fucken works!!!
# Out of caution, lets generate more data.
# Also, in the optimization problem, let's repeat the test for 20 times, and take the model with the highest score.