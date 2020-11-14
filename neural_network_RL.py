import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return x * (1 - x)


def feed_forward(inp, weights):
    return np.array([sigmoid(x) for x in np.dot(inp, weights)])


class NeuralNetwork:
    def __init__(self, input_len, output_len, hidden_layer_len=0, learning_rate=1):
        self.weights1 = (np.random.rand(input_len, hidden_layer_len) - 0.5) / (2 * input_len)
        self.weights2 = (np.random.rand(hidden_layer_len, output_len) - 0.5) / (2 * hidden_layer_len)
        self.alpha = learning_rate

    def backpropagation(self, inputs, hidden_layer, outputs, loss):
        if outputs[0] - 0.5 > 0:
            if loss > 0:
                loss = loss * (1 - outputs[0])
            else:
                loss = loss * outputs[0]
        else:
            if loss > 0:
                loss = - loss * outputs[0]
            else:
                loss = - loss * (1 - outputs[0])
        delta = loss * sigmoid_der(outputs)
        error1 = np.outer(delta, self.weights2)
        delta1 = error1 * sigmoid_der(hidden_layer)
        self.weights2 += self.alpha * np.dot(hidden_layer.T, delta)
        self.weights1 += np.dot(inputs.T, delta1)

    def train(self, inputs, loss):
        hidden_layer = feed_forward(inputs, self.weights1)
        outputs = feed_forward(hidden_layer, self.weights2)
        self.backpropagation(inputs, hidden_layer, outputs, loss)

    def get_result(self, inputs):
        hidden_layer = feed_forward(inputs, self.weights1)
        outputs = feed_forward(hidden_layer, self.weights2)
        return outputs

