from neural_net import NeuralNetwork
import numpy as np
import matplotlib.pyplot
from datetime import datetime as dt

# configure and initialize neural network
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1  # 0.3 is .94, 0.1 is .9523, 0.15 is .9527, 0.2 is .9511
training_epochs = 7

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print("training neural network...")
train_start = dt.now()

# read training data
with open("data/mnist_train.csv", "r") as f:
    train_data = f.readlines()

# train neural network
for _ in range(training_epochs):
    for r in train_data:
        vals = r.split(",")
        input = (np.asfarray(vals[1:]) / 255.0 * 0.99) + 0.01
        # img_array = np.asfarray(input).reshape((28,28))
        # matplotlib.pyplot.imshow(img_array, cmap="Greens", interpolation="None")
        # matplotlib.pyplot.show()
        target = np.zeros(output_nodes) + 0.01
        target[int(vals[0])] = 0.99
        n.train(input, target)

test_start = dt.now()
print("training complete")
print(f"train time: {test_start - train_start}s")
print("testing neural network...")

# test neural network
with open("data/mnist_test.csv", "r") as f:
    test_data = f.readlines()

scorecard = []
for r in test_data:
    vals = r.split(",")
    correct_ans = int(vals[0])
    input = (np.asfarray(vals[1:]) / 255.0 * 0.99) + 0.01
    res = np.argmax(n.query(input))
    # print(f"correct answer: {correct_ans}")
    # print(f"predicted result: {np.argmax(res) + 1}\n")
    if res == correct_ans:
        scorecard.append(1)
    else:
        scorecard.append(0)
end = dt.now()
print("testing complete")
print(f"test time: {end - test_start}s\n")
# print(scorecard)

# calculate performance score
print("Results:")
scorecard_array = np.asarray(scorecard)
print(f"performance = {scorecard_array.sum() / scorecard_array.size}")
print(f"learning rate: {learning_rate}")
print(f"epochs: {training_epochs}")
