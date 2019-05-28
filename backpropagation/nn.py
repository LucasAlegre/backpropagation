import numpy as np 


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


class NN:

    def __init__(self, architecture, regularization_factor=0, initial_weights=None, alpha=0.001):
        if type(architecture) is str:
            self._build_architecture_from_file(architecture)
        else:
            self.architecture = architecture
            self.regularization_factor = regularization_factor

        self.activations = []
        self.weights = []
        if initial_weights is None:
            self._init_random_weights()
        else:
            self._read_weights_from_file(initial_weights)
        
        self.alpha = alpha
    
    def train(self, x, y):
        pass

    def predict(self, instace):
        pass

    def propagate(self, x):
        sigmoid_vec = np.vectorize(sigmoid)
        activations = [np.append(1.0, x).reshape(-1,1)]
        for layer in range(1, self.num_layers-1):
            z = np.dot(self.weights[layer-1], activations[layer-1])
            a = np.append(1.0, sigmoid_vec(z)).reshape(-1,1)
            activations.append(a)
        z = np.dot(self.weights[self.num_layers-2], activations[self.num_layers-2])
        activations.append(sigmoid_vec(z))
        self.activations = activations
        print(self.activations)
        return self.activations[self.num_layers-1]

    def _build_architecture_from_file(self, architecture):
        with open(architecture, 'r') as f:
            fileread = f.read().splitlines()
        self.regularization_factor = float(fileread[0])
        self.architecture = [int (x) for x in fileread[1:]]

    @property
    def num_layers(self):
        return len(self.architecture)
    
    def _init_random_weights(self):
        for layer in range(self.num_layers-1):
            self.weights.append(np.random.normal(size=(self.architecture[layer+1], self.architecture[layer]+1)))

    def _read_weights_from_file(self, file):
        w = []
        with open(file, 'r') as f:
            for line in f:
                w.append([float(x) for x in line.replace(';',',').split(',')])
        for layer in range(self.num_layers-1):
            self.weights.append(np.asmatrix(w[layer]).reshape(self.architecture[layer+1], self.architecture[layer]+1))

    def view_architecture(self):
        pass
