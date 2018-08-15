import numpy as np
class NeuralNet:
    def __init__(self, num_layers, num_nodes, activation_func):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.act_func = activation_func
        self.layers = []
        for i in range(num_layers):
            if i != num_layers - 1:
                layer_i = Layers(num_nodes[i], num_nodes[i + 1], activation_func[i])
            else:
                layer_i = Layers(num_nodes[i], 0, activation_func[i])
            self.layers.append(layer_i)
    def forwardprop(self, input):
        self.layers[0].activations = input
        for i in range(self.num_layers-1):
            temp = np.add(self.layers[i].bias_for_layer, np.matmul(self.layers[i].activations, self.layers[i].weights_for_layer))
            if i == 0:
                self.layers[i+1].activations = self.relu(temp)
            elif i == 1:
                self.layers[i+1].activations = self.softmax(temp)
            elif i == 2:
                self.layers[i+1].activations = self.sigmoid(temp)

    def relu(self, layer):
        layer[layer < 0] = 0
        return layer

    def softmax(self, layer):
        exp = np.exp(layer)
        if isinstance(layer[0], np.ndarray):
            return exp/np.sum(exp, axis=1, keepdims=True)
        else:
            return exp/np.sum(exp, keepdims=True)

    def sigmoid(self, layer):
        return np.divide(1, np.add(1, np.exp(np.negative(layer))))

    def calculateerror(self, labels):
        self.error += np.mean(
            np.divide(np.square(np.subtract(labels, self.layers[self.num_layers - 1].activations)), 2))

class Layers:
    def __init__(self, nodes_in_layer, nodes_in_next_layer, activation_func):
        self.nodes = nodes_in_layer
        self.next_nodes = nodes_in_next_layer
        self.activation_func = activation_func
        self.activations = np.zeros([self.nodes, 1])
        if nodes_in_next_layer != 0:
            self.weights_for_layer = np.random.normal(0, 0.001, size=(self.nodes, self.next_nodes))
            self.bias_for_layer = np.random.normal(0, 0.001, size=(1, self.next_nodes))
        else:
            self.weights_for_layer = None
            self.bias_for_layer = None
