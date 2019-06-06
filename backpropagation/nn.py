import numpy as np 


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


class NN:

    def __init__(self, architecture, regularization_factor=0, initial_weights=None, alpha=0.001, verbose=False, momentum=True, beta=0.9):
        """Feedforward Neural Network
        
        Args:
            architecture (list/str): List of number of neurons per layer or string with .txt file.
            regularization_factor (int, optional): Regularization Factor. Defaults to 0.
            initial_weights (str, optional): txt file with initial weights. If None, weights are sampled from N(0,1). Defaults to None.
            alpha (float, optional): Learning rate. Defaults to 0.001.
            momentum (boolean, optional): Use or not the Momentum Method to correct weights. Defaults to True.
            beta (float, optional): Efective direction rate used on the Momentum Method. Defaults to 0.9.
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
        self.verbose = verbose
        self.momentum = momentum
        if momentum:
            self.beta = beta
            self.init_z_directions()
            self.apply_grads = self.apply_grads_with_momentum_method
        else:
            self.apply_grads = self.apply_grads_with_usual_method
    
    def predict(self, instace):
        pass
    
    def train(self, x, y):
        num_examples = len(x)
        for _ in range(50):
            sum_loss = 0.0
            for i in range(num_examples):
                xi, yi = x[i], y[i].reshape(-1,1)
                fx = self.propagate(xi)
                loss = self.cost(fx, yi)
                sum_loss += loss
                self.backpropagate(fx, yi)
            mean_loss = (sum_loss/num_examples)
            regularized_loss =  mean_loss + self.regularization_cost(num_examples)
            self.add_regularization_to_grads(num_examples)
            self.apply_grads()
    
            print('Total loss: ', regularized_loss)

    def backpropagate(self, fx, y):
        """Computes gradients using backpropagation
        Args:
            fx (np.array): NN predictions
            y (np.array): True outputs
        """
        self.calculate_deltas(fx, y)
        self.calcutate_grads()

    def calculate_deltas(self, fx, y):
        # Set output layer separately
        np.subtract(fx, y, out=self.deltas[-1])
        for layer in range(self.num_layers-2, 0, -1):
            weights = self.weights[layer].transpose()[1:]
            activations_element_by_element = np.multiply(self.activations[layer][1:], (1 - self.activations[layer][1:]))
            weights_dot_product_deltas = np.dot(weights, self.deltas[layer])
            np.multiply(activations_element_by_element, weights_dot_product_deltas, out=self.deltas[layer-1])

    def propagate(self, x):
        """Propagates forward an instance, computing the activation of each neuron
        Args:
            x (np.array): Instance
        Returns:
            np.array: The output layer return
        """
        np.copyto(self.activations[0], np.append(1.0, x).reshape(-1,1))
        for layer in range(1, self.num_layers-1):
            np.dot(self.weights[layer-1], self.activations[layer-1], out=self.activations[layer][1:])
            self.activations[layer] = sigmoid(self.activations[layer])
            self.activations[layer][0][0] = 1.0
        np.dot(self.weights[self.num_layers-2], self.activations[self.num_layers-2], out=self.activations[self.num_layers-1])
        self.activations[self.num_layers-1] = sigmoid(self.activations[self.num_layers-1])
        return self.activations[self.num_layers-1]

    def cost(self, fx, y):
        """Cross-entropy loss
        Args:
            fx (np.array): NN instance predictions
            y (np.array): True outputs
        Returns:
            float: Logistic loss
        """
        j = 0.0
        for i in range(self.architecture[-1]):
            j += -((y[i] * np.log(fx[i]) + (1 - y[i]) * np.log(1 - fx[i])))
        return j

    def regularization_cost(self, num_examples):
        return (self.regularization_factor/(2*num_examples)) * np.sum(np.square(w[:, 1:]).sum() for w in self.weights)

    def calcutate_grads(self):
        """Computes the gradient for each weight
        """
        for i in range(self.num_layers-2, -1, -1):
            grad = np.dot(self.deltas[i], self.activations[i].reshape(1,-1))
            self.grads[i] += grad

    def apply_grads_with_usual_method(self):
        grads_with_learning_rate = np.multiply(self.grads, self.alpha)
        self.weights = np.subtract(self.weights, grads_with_learning_rate)

    def apply_grads_with_momentum_method(self):
        self.z_directions = np.multiply(self.z_directions, self.beta)
        self.z_directions = np.add(self.z_directions, self.grads)
        grads_with_learning_rate = np.multiply(self.z_directions, self.alpha)
        self.weights = np.subtract(self.weights, grads_with_learning_rate)
    
    def add_regularization_to_grads(self, num_examples):
        """Sums the regularization times the weights to the gradients, and computes the mean gradient
        Args:
            num_examples (int): Number of instances used to compute the gradient
        """
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

    def init_z_directions(self):
        self.z_directions = [np.zeros(layer.shape) for layer in self.weights]

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
