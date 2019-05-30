import numpy as np 


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


class NN:

    def __init__(self, architecture, regularization_factor=0, initial_weights=None, alpha=0.001):
        """Feedforward Neural Network
        
        Args:
            architecture (list/str): List of number of neurons per layer or string with .txt file.
            regularization_factor (int, optional): Regularization Factor. Defaults to 0.
            initial_weights (str, optional): txt file with initial weights. If None, weights are sampled from N(0,1). Defaults to None.
            alpha (float, optional): Learning rate. Defaults to 0.001.
        """
        if type(architecture) is str:
            self._build_architecture_from_file(architecture)
        else:
            self.architecture = architecture
            self.regularization_factor = regularization_factor

        self._init_activations()
        if initial_weights is None:
            self._init_random_weights()
        else:
            self._read_weights_from_file(initial_weights)
        
        self.alpha = alpha
    
    def train(self, x, y):
        print(self.cost(x,y))

    def predict(self, instace):
        pass

    def propagate(self, x):
        np.copyto(self.activations[0], np.append(1.0, x).reshape(-1,1))
        for layer in range(1, self.num_layers-1):
            np.dot(self.weights[layer-1], self.activations[layer-1], out=self.activations[layer][1:])
            self.activations[layer] = sigmoid(self.activations[layer])
            self.activations[layer][0][0] = 1.0
        np.dot(self.weights[self.num_layers-2], self.activations[self.num_layers-2], out=self.activations[self.num_layers-1])
        self.activations[self.num_layers-1] = sigmoid(self.activations[self.num_layers-1])
        return self.activations[self.num_layers-1]

    def cost(self, x, y):
        n = len(x)
        j = 0.0
        for i in range(n):
            fx = self.propagate(x[i])
            j += (-y[i] * np.log(fx) - (1 - y[i]) * np.log(1 - fx)).sum()
        j = j/n
        s = sum(np.square(w).sum() for w in self.weights) * (self.regularization_factor/(2*n))
        return j + s

    def _build_architecture_from_file(self, architecture):
        with open(architecture, 'r') as f:
            fileread = f.read().splitlines()
        self.regularization_factor = float(fileread[0])
        self.architecture = [int (x) for x in fileread[1:]]

    @property
    def num_layers(self):
        return len(self.architecture)
    
    def _init_activations(self):
        self.activations = []
        for layer in range(self.num_layers-1):
            self.activations.append(np.empty((self.architecture[layer]+1,1)))  # column vector with bias
            self.activations[layer][0][0] = 1.0  # bias neuron
        self.activations.append(np.empty((self.architecture[-1],1))) # output layer doesn't have bias
    
    def _init_random_weights(self):
        self.weights = []
        for layer in range(self.num_layers-1):
            self.weights.append(np.random.normal(size=(self.architecture[layer+1], self.architecture[layer]+1)))

    def _read_weights_from_file(self, file):
        self.weights = []
        w = []
        with open(file, 'r') as f:
            for line in f:
                w.append([float(x) for x in line.replace(';',',').split(',')])
        for layer in range(self.num_layers-1):
            self.weights.append(np.asmatrix(w[layer]).reshape(self.architecture[layer+1], self.architecture[layer]+1))

    def view_architecture(self):
        pass
