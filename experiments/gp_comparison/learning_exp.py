__author__ = 'omohamme'
######################################################################
# - For 2-D (which we know is OK to learn), do the following:
# 	- Make a new simulation for a damaged robot (remove one leg).
# 	- Modify the "test_case_service" script to accept a new input to decide if the test is on a
# 	damaged robot or a good one.
# 	- On the good robot, learn the following models and score them:
# 		- model 0 : (svm_prediction + gp_prediction) / 2
# 		- model 1 : svm_prediction * gp_prediction
# 		- model 2 : svm_prediction + robot_good_gp_model mean square error for the predicted values
# 		- model 3 : svm alone
# 		- model 4 : gp_prediction alone
# 	- Save these models to an external file.
# 	- On the damaged robot, get 1000 random samples, and score each of the previously
# 	learned models (from the good robot) with these points.

# EXP2: What I need to test is:
# 1. Learn SVM from the good robot, and test its performance.
# 2. Now, select some points from the damaged robot.
# 3. Predict their performance with the SVM.
# 4. Measure the differences between the predicted points and the
######################################################################
from utilities import *
load_from_file = False
clear_screen()
dimensions = 6
if not load_from_file:
    models_scores_good_robot = dict(#svm_model=[],
                        robot_good_DecisionTree_model=[],
                        robot_good_DecisionTreePLUSgp_model=[],
                        # svm_plus_gp_model=[],
                        # svm_plus_gpnoise_model=[],
                        # svm_mul_gp_model=[],
                         )

    models_scores_damage_robot = dict(#svm_model=[],
                        robot_good_DecisionTree_model=[],
                        robot_good_DecisionTreePLUSgp_model=[],
                        # svm_plus_gp_model=[],
                        # svm_plus_gpnoise_model=[],
                        # svm_mul_gp_model=[],
                         )
    objectives_list = [1]
    for i in range(50): #Repeat this exp 100 times
        print "Run Number : ", i
        # Initialize the basic models for the good robot
        # robot_good_DecisionTree_model = robot_learning(num_in_samples=100000, objectives_list=objectives_list, learninig_model=DecisionTreeRegressor(max_depth=40),
        #                                       robot_condition="good", dimension=dimensions)
        robot_good_DecisionTree_model = robot_learning(num_in_samples=40000, objectives_list=objectives_list, learninig_model=DecisionTreeRegressor(max_depth=40),
                                              robot_condition="ONLYGOOD", dimension=dimensions)

        robot_good_DecisionTree_model.get_training_data()
        robot_good_DecisionTree_model.fit_learning_model()

        # Measure the current model performance
        test_method = robot_test_samples(num_test_samples=1000, objectives_list=objectives_list, robot_condition="ONLYGOOD", dimension=dimensions)
        test_method.get_test_samples()
        test_points, test_results = test_method.return_test_pairs()

        model_0 = robot_good_DecisionTree_model.predict(test_points)

        models_scores_good_robot["robot_good_DecisionTree_model"].append(repr(r2_score(test_results[0], model_0)))

        # Build a new test method for the damaged robot!
        test_method = robot_test_samples(num_test_samples=200, objectives_list=objectives_list, robot_condition="ONLYDAMAGE", dimension=dimensions)
        test_method.get_test_samples()
        test_points, test_results = test_method.return_test_pairs()

        model_0 = robot_good_DecisionTree_model.predict(test_points) + robot_good_DecisionTree_model.mean_value[0]
        models_scores_damage_robot["robot_good_DecisionTree_model"].append(repr(r2_score(test_results[0]+test_method.mean_value[0], model_0)))

        gp_damaged_robot_diff = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget=3.00e-13)
        difference = ((test_results + test_method.mean_value[0]) - model_0)[0]
        gp_damaged_robot_diff.fit(test_points, difference)

        #Now, test the combined model

        test_method = robot_test_samples(num_test_samples=1000, objectives_list=objectives_list, robot_condition="ONLYDAMAGE", dimension=dimensions)
        test_method.get_test_samples()
        test_points, test_results = test_method.return_test_pairs()

        model_0 = np.asarray(robot_good_DecisionTree_model.predict(test_points) + robot_good_DecisionTree_model.mean_value[0]) + gp_damaged_robot_diff.predict(test_points)
        models_scores_damage_robot["robot_good_DecisionTreePLUSgp_model"].append(repr(r2_score(test_results[0]+test_method.mean_value[0], model_0)))

        print models_scores_good_robot
    with open("learning_exp_good_robot.json", 'w') as fp:
        json.dump(models_scores_good_robot, fp)

    with open("learning_exp_damage_robot.json", 'w') as fp:
        json.dump(models_scores_damage_robot, fp)

else:
    with open("learning_exp_good_robot.json", 'r') as fp:
        models_scores_good_robot = json.loads(fp.readline())

    with open("learning_exp_damage_robot.json", 'r') as fp:
        models_scores_damage_robot = json.loads(fp.readline())

# combined_results = {}
combined_results = []
labels = []
for item in models_scores_good_robot:
    # combined_results[item + "_good"] = map(float, models_scores_good_robot[item])
    # combined_results[item + "_damage"] = map(float, models_scores_damage_robot[item])
    combined_results.append(map(float, models_scores_good_robot[item]))
    combined_results.append(map(float, models_scores_damage_robot[item]))
    labels.append(item + "_good")
    labels.append(item + "_damage")

current_path = os.getcwd()
plt.figure(figsize=(15.0, 11.0))
# plt.boxplot(combined_results.values())
# plt.xticks(range(1, len(combined_results.keys()) + 1), combined_results.keys(), rotation = 20, fontsize=10)
plt.boxplot(combined_results)
plt.xticks(range(1, len(labels) + 1), labels, rotation = 20, fontsize=10)
plt.yticks(fontsize=10)
# plt.ylim([0, 1])
plt.ylabel("R2 Score", fontsize=10)
plt.xlabel("Different models", fontsize=10)
# plt.show()
# plt.savefig("learning_exp_results_V2controller_height.png")
plt.savefig("learning_exp_results_speed.png")
os.chdir(current_path)