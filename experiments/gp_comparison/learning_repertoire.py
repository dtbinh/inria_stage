__author__ = 'omohamme'
from utilities import *

with open("archive.dat", "r") as file:
    repertoire_text = file.read().splitlines()

data_set = {}
all_data_input = []
all_data_output = []
for line in repertoire_text:
    line_split = line.split("    ")
    input_data = map(float, line_split[0].split(" "))
    output_data = float(line_split[1])
    all_data_input.append(input_data)
    all_data_output.append(output_data)

# Get the data mean
data_mean = sum(all_data_output) / len(all_data_output)

all_model_score = []

for j in range(10):
    model_score = []
    model_score_1 = []
    print "Current iteration =====> ", j
    for i in range(10):
        print i
        # Get the training sample
        training_input = []
        training_output = []
        random_indices = random.sample(range(0, len(all_data_input)), 20)
        for index in random_indices:
            training_input.append(all_data_input[index])
            training_output.append(all_data_output[index] - data_mean)

        # Train the GP
        # gp_model = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget=3.00e-13)
        gp_model = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
        gp_model.fit(training_input, training_output)

        gp_model_1 = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget=3.00e-13)
        gp_model_1.fit(training_input, training_output)

        # Now, get test data
        test_input = []
        test_output = []
        random_indices = random.sample(range(0, len(all_data_input)), 1000)
        for index in random_indices:
            test_input.append(all_data_input[index])
            test_output.append(all_data_output[index] - data_mean)

        model_score.append(gp_model.score(test_input, test_output))
        model_score_1.append(gp_model_1.score(test_input, test_output))

    all_model_score.append(model_score)
plt.figure(figsize=(15.0, 11.0))
# plt.boxplot(combined_results.values())
# plt.xticks(range(1, len(combined_results.keys()) + 1), combined_results.keys(), rotation = 20, fontsize=10)
# plt.boxplot([model_score, model_score_1])
plt.boxplot(all_model_score)
plt.ylabel("R2 Score", fontsize=10)
plt.xlabel("Different models", fontsize=10)
# plt.show()
# plt.savefig("learning_exp_results_V2controller_height.png")
plt.savefig("learning_REPERTOIRE.png")
