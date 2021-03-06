__author__ = 'omohamme'
# Note here that I don't fix the mean of the gaussian process on the data center, but just on zero (the default configuration). This is not
# right, but doesn't have much effect on the results -- To be fixed later.
from utilities import *

for test_num in range(50):
    dt_model = DecisionTreeRegressor(max_depth=40)
    # dt_model = svm.SVR()
    training_samples, training_results = data_parser(num_samples=30000, controller_dim=6, robot_condition="ONLYGOOD")

    dt_model.fit(training_samples, training_results[0])

    test_samples, test_results = data_parser(num_samples=1000, controller_dim=6, robot_condition="ONLYGOOD")

    model_score = dt_model.score(test_samples, test_results[0])

    print "DT model score on the good robot = ", model_score
    # Now, train a GP model on the difference between DT and the actual values
    # gp_model = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget=3.00e-13)
    gp_model = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
    test_samples, test_results = data_parser(num_samples=200, controller_dim=6, robot_condition="ONLYDAMAGE1")
    model_score = dt_model.score(test_samples, test_results[0])
    dt_damage_model_score = dt_model.score(test_samples, test_results[0])
    dt_damage_model_pred = dt_model.predict(test_samples)

    print "DT model score on the damaged robot = ", dt_damage_model_score

    diff_pred_real = dt_damage_model_pred - test_results[0]

    gp_model.fit(test_samples, diff_pred_real)

    #Now, test the new model
    test_samples, test_results = data_parser(num_samples=1000, controller_dim=6, robot_condition="ONLYDAMAGE1")
    dt_damage_model_pred = dt_model.predict(test_samples)
    gp_damage_model_pred = gp_model.predict(test_samples)
    print "COMBINED model score on the damaged robot PLUS = ", repr(r2_score(test_results[0], dt_damage_model_pred+gp_damage_model_pred))
    print "COMBINED model score on the damaged robot MIN  = ", repr(r2_score(test_results[0], dt_damage_model_pred-gp_damage_model_pred))

    # print diff_pred_real[0:20]
    # print gp_damage_model_pred[0:20]
    print "-------------------------------------------------"
    # exit()
