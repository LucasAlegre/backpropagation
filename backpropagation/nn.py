import numpy as np 
from backpropagation.util import to_one_hot


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


class NN:

    def __init__(self, architecture, regularization_factor=0, initial_weights=None, alpha=0.001, verbose=False,
                       momentum=True, beta=0.9, class_column=None, class_values=None, epochs=100, batch_size=1):
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

        self.initial_weights = initial_weights
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.momentum = momentum
    
        self.reset()

        if self.momentum:
            self.beta = beta
            self.apply_grads = self.apply_grads_with_momentum_method
        else:
            self.apply_grads = self.apply_grads_with_usual_method

        self.class_column = class_column
        self.class_values = class_values

    def reset(self):
        self.init_activations()
        self.init_deltas()
        self.init_grads()
        self.init_weights()

        if self.momentum:
            self.init_z_directions()
    
    def predict(self, instance):
        instance = instance.drop(labels=[self.class_column]).values
        instance = np.array(instance, dtype='float64')
        activations = list(self.propagate(instance))
        max_index = activations.index(max(activations))
        return self.class_values[max_index]
    
    def train(self, data):
        self.reset()
        x = data.drop(self.class_column, axis=1).values
        y = to_one_hot(data[self.class_column])
        n = len(x)
        batches_x = np.array_split(x, self.batch_size)
        batches_y = np.array_split(y, self.batch_size)
        for e in range(self.epochs):
            epoch_loss = 0.0
            for batch in range(self.batch_size):
                sum_loss = 0.0
                self.reset_grads()
                true_batch_size = len(batches_x[batch])

                for i in range(true_batch_size):
                    xi, yi = batches_x[batch][i], batches_y[batch][i].reshape(-1,1)
                    fx = self.propagate(xi)
                    sum_loss += self.cost(fx, yi)
                    self.backpropagate(fx, yi)

                mean_loss = sum_loss / true_batch_size
                regularized_loss = mean_loss + self.regularization_cost(true_batch_size)
                #print('Batch ' + str(batch+1) + ' - Total loss: ', regularized_loss)
                epoch_loss += sum_loss
                self.add_regularization_to_grads(true_batch_size)
                self.apply_grads()
    
            print('Epoch ' + str(e+1) + ' - Total loss: ', epoch_loss/n + self.regularization_cost(n))

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
        j = -((y * np.log(fx) + (1 - y) * np.log(1 - fx)))
        return j.sum()

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
        self.weights -= grads_with_learning_rate

    def apply_grads_with_momentum_method(self):
        self.z_directions = np.multiply(self.z_directions, self.beta)
        self.z_directions = np.add(self.z_directions, self.grads)
        grads_with_learning_rate = np.multiply(self.z_directions, self.alpha)
        self.weights -= grads_with_learning_rate
    
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
    
    def reset_grads(self):
        for g in self.grads:
            g.fill(0.0)
        if self.momentum:
            for z in self.z_directions:
                z.fill(0.0)

    def init_activations(self):
        self.activations = []
        for layer in range(self.num_layers-1):
            self.activations.append(np.empty((self.architecture[layer]+1,1)))  # column vector with bias
            self.activations[layer][0][0] = 1.0  # bias neuron
        self.activations.append(np.empty((self.architecture[-1],1))) # output layer doesn't have bias

    def init_weights(self):
        if self.initial_weights is None:
            self.init_random_weights()
        else:
            self.read_weights_from_file(self.initial_weights)
    
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
