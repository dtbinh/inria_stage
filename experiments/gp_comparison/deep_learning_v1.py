from utilities import *
import data_parser_mathew

num_iteration = [10, 100, 1000, 10000, 50000, 100000, 200000]
# num_iteration = [10, 20, 30]
score_list = []
for i in num_iteration:
    input_data, output_data = data_parser_mathew.data_parser(num_in_samples=70000)
    test_input_data, test_output_data = data_parser_mathew.data_parser(num_in_samples=1000)
    svm_model = DecisionTreeRegressor(max_depth=20)
    rbm = BernoulliRBM(random_state=0, verbose=True)
    classifier = Pipeline(steps=[('rbm', rbm), ('svm', svm_model)])
    rbm.learning_rate = 0.001
    rbm.n_iter = i
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 8

    classifier.fit(input_data, output_data)

    model_score = repr(r2_score(test_output_data, classifier.predict(test_input_data)))

    print model_score

    score_list.append(model_score)

score_list = map(float, score_list)
plt.figure()
plt.plot(num_iteration, score_list)
plt.ylabel("R2 score")
plt.xlabel("Number of iteration")
# plt.boxplot(score_list)
plt.show()