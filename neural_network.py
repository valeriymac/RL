import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return x * (1 - x)


def feed_forward(inp, weights):
    return np.array([sigmoid(x) for x in np.dot(inp, weights)])


class NeuralNetwork:
    def __init__(self, input_len, output_len, hidden_layer_len=0):
        self.weights1 = (np.random.rand(input_len, hidden_layer_len) - 0.5) / (2 * input_len)
        self.weights2 = (np.random.rand(hidden_layer_len, output_len) - 0.5) / (2 * hidden_layer_len)

    def backpropagation(self, inputs, hidden_layer, outputs, ideal_outputs):
        loss1 = ideal_outputs - outputs
        delta1 = loss1 * sigmoid_der(outputs)
        loss2 = np.outer(delta1, self.weights2)
        delta2 = loss2 * sigmoid_der(hidden_layer)
        self.weights2 += np.dot(hidden_layer.T, delta1)
        self.weights1 += np.dot(inputs.T, delta2)

    def train(self, inputs, ideal_outputs):
        hidden_layer = feed_forward(inputs, self.weights1)
        outputs = feed_forward(hidden_layer, self.weights2)
        self.backpropagation(inputs, hidden_layer, outputs, ideal_outputs)

    def get_result(self, inputs):
        hidden_layer = feed_forward(inputs, self.weights1)
        outputs = feed_forward(hidden_layer, self.weights2)
        return outputs


inpp = np.random.rand(10000, 3)
inp1 = np.empty([10000, 3], dtype=float)
out1 = np.empty((10000, 1))
for n in range(10000):
    out1[n] = np.array([1])
    for i in range(3):
        inp1[n][i] = round(inpp[n][i])
        if inp1[n][i] == 0:
            out1[n] = np.array([0])

a = NeuralNetwork(3, 1, 20)
for n in range(10000):
    print(np.array([inp1[n]]), np.array([out1[n]]))
    a.train(np.array([inp1[n]]), np.array([out1[n]]))
print(a.get_result(np.array([[0., 1., 1.]])))
print(a.get_result(np.array([[1., 1., 1.]])))
