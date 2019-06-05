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
            self.build_architecture_from_file(architecture)
        else:
            self.architecture = architecture
            self.regularization_factor = regularization_factor

        self.init_activations()
        self.init_deltas()
        self.init_grads()
        if initial_weights is None:
            self.init_random_weights()
        else:
            self.read_weights_from_file(initial_weights)
    
        self.alpha = alpha
    
    def predict(self, instace):
        pass
    
    def train(self, x, y):
        num_examples = len(x)
        sum_loss = 0.0

        for i in range(num_examples):
            fx = self.propagate(x[i])
            loss = self.cost(fx, y[i])
            sum_loss += loss
            print('Example {} loss: {}'.format(i, loss))
            self.backpropagate(fx, y[i])

        regularized_loss = (sum_loss/num_examples) + np.sum(np.square(w).sum() for w in self.weights) * (self.regularization_factor/(2*num_examples))
        self.add_regularization_to_grads(num_examples)
        self.apply_grads()
        print('Total loss: ', regularized_loss)
        print('Gradients:', self.grads)

    def backpropagate(self, fx, y):
        self.calculate_deltas(fx, y)
        self.calcutate_grads()

    def calculate_deltas(self, fx, y):
        # Set output layer separately
        self.deltas[-1] = fx - y
        for layer in range(self.num_layers-2, 0, -1):
            weights = self.weights[layer].reshape(-1, 1)[1:]
            activations_element_by_element = np.multiply(self.activations[layer][1:], (1 - self.activations[layer][1:]))
            weights_dot_product_deltas = weights * self.deltas[layer]
            np.multiply(activations_element_by_element, weights_dot_product_deltas, out=self.deltas[layer - 1])

    def propagate(self, x):
        np.copyto(self.activations[0], np.append(1.0, x).reshape(-1,1))
        for layer in range(1, self.num_layers-1):
            np.dot(self.weights[layer-1], self.activations[layer-1], out=self.activations[layer][1:])
            self.activations[layer] = sigmoid(self.activations[layer])
            self.activations[layer][0][0] = 1.0
        np.dot(self.weights[self.num_layers-2], self.activations[self.num_layers-2], out=self.activations[self.num_layers-1])
        self.activations[self.num_layers-1] = sigmoid(self.activations[self.num_layers-1])
        return self.activations[self.num_layers-1]

    def cost(self, fx, y):
        n = len(fx)
        j = 0.0
        for i in range(n):
            j += (-y[i] * np.log(fx[i]) - (1 - y[i]) * np.log(1 - fx[i])).sum()
        return j

    def calcutate_grads(self):
        for i in range(self.num_layers-2, -1, -1):
            grad = np.dot(self.deltas[i], self.activations[i].reshape(1,-1))
            self.grads[i] += grad

    def apply_grads(self):
        pass
    
    def add_regularization_to_grads(self, num_examples):
        for i in range(self.num_layers-2, -1, -1):
            p = self.weights[i].copy()
            p[:, 0] = 0  # ignore bias weights
            self.grads[i] += self.regularization_factor * p
            self.grads[i] /= num_examples

    @property
    def num_layers(self):
        return len(self.architecture)
    
    def init_activations(self):
        self.activations = []
        for layer in range(self.num_layers-1):
            self.activations.append(np.empty((self.architecture[layer]+1,1)))  # column vector with bias
            self.activations[layer][0][0] = 1.0  # bias neuron
        self.activations.append(np.empty((self.architecture[-1],1))) # output layer doesn't have bias
    
    def init_random_weights(self):
        self.weights = []
        for layer in range(self.num_layers-1):
            self.weights.append(np.random.normal(size=(self.architecture[layer+1], self.architecture[layer]+1)))

    def read_weights_from_file(self, file):
        self.weights = []
        w = []
        with open(file, 'r') as f:
            for line in f:
                w.append([float(x) for x in line.replace(';',',').split(',')])
        for layer in range(self.num_layers-1):
            self.weights.append(np.asmatrix(w[layer]).reshape(self.architecture[layer+1], self.architecture[layer]+1))

    def init_deltas(self):
        self.deltas = [np.empty((self.architecture[n], 1)) for n in range(1, self.num_layers)]

    def init_grads(self):
        self.grads = []
        for layer in range(self.num_layers-1):
            self.grads.append(np.zeros((self.architecture[layer+1], self.architecture[layer]+1)))

    def build_architecture_from_file(self, architecture):
        with open(architecture, 'r') as f:
            fileread = f.read().splitlines()
        self.regularization_factor = float(fileread[0])
        self.architecture = [int (x) for x in fileread[1:]]

    def view_architecture(self):
        pass
