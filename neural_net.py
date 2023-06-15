# Author: Nicholas Debeurre

import numpy as np
import scipy.special as sp


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # nodes
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # weights
        # simple base method
        # self.input2hidden_w = (np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        # self.hidden2output_w = (np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)
        # normalized distribution method accounting for std dev of num incoming links
        self.input2hidden_w = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.hidden2output_w = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # define the learning rate
        self.lr = learning_rate

        # activation function
        self.activation_function = lambda x: sp.expit(x)

    def train(self, inputs_list, targets_list):
        # convert inputs and targets to 2d arrays
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.input2hidden_w, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        output_inputs = np.dot(self.hidden2output_w, hidden_outputs)
        output_outputs = self.activation_function(output_inputs)

        # calculate error
        output_errors = targets - output_outputs
        hidden_errors = np.dot(self.hidden2output_w.T, output_errors)

        # update weights
        self.hidden2output_w += self.lr * np.dot((output_errors * output_outputs * (1.0 - output_outputs)), np.transpose(hidden_outputs))
        self.input2hidden_w += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs):
        # convert inputs to 2d array
        input = np.array(inputs, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.input2hidden_w, input)

        # calculate signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        output_inputs = np.dot(self.hidden2output_w, hidden_outputs)

        # calculate signals from final output layer
        output_outputs = self.activation_function(output_inputs)

        return output_outputs
