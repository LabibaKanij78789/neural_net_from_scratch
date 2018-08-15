import numpy as np
class NeuralNet:
    def __init__(self, num_layers, num_nodes, activation_func, cost_func):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.act_func = activation_func
        self.cost_func = cost_func
        self.layers = []
        for i in range(num_layers):
            if i != num_layers - 1:
                layer_i = Layers(num_nodes[i], num_nodes[i + 1], activation_func[i])
            else:
                layer_i = Layers(num_nodes[i], 0, activation_func[i])
            self.layers.append(layer_i)


class Layers:
    def __init__(self, nodes_in_layer, nodes_in_next_layer, activation_func):
        self.nodes = nodes_in_layer
        self.next_nodes = nodes_in_next_layer
        self.activation = activation_func
        if nodes_in_next_layer != 0:
            self.weights_for_layer = np.random.normal(0, 0.001, size=(self.nodes, self.next_nodes))
            self.bias_for_layer = np.random.normal(0, 0.001, size=(1, self.next_nodes))
        else:
            self.weights_for_layer = None
            self.bias_for_layer = None
