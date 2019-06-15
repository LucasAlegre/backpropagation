import numpy as np 
from tqdm import tqdm
from backpropagation.util import to_one_hot
import copy


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


class NN:

    def __init__(self, architecture, regularization_factor=0.0, initial_weights=None, alpha=0.001, optimizer='Adam', 
                 beta=0.9, class_column=None, class_values=None, epochs=100, batch_size=None):
        """Feedforward Neural Network
        
        Args:
            architecture (list/str): List of number of neurons per layer or string with .txt file.
            regularization_factor (int, optional): Regularization Factor. Defaults to 0.
            initial_weights (str, optional): txt file with initial weights. If None, weights are sampled from N(0,1). Defaults to None.
            alpha (float, optional): Learning rate. Defaults to 0.001.
            optimizer (str, optional): Which optimizer to user. ['SGD', 'Momentum', 'Adam']
            beta (float, optional): Efective direction rate used on the Momentum Method. Defaults to 0.9.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            batch_size (int, optional): Size of mini-batch. Defaults to size of the complete dataset.
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
        self.optimizer = optimizer
    
        self.reset()

        if self.optimizer == 'Momentum':
            self.beta = beta
            self.apply_grads = self.apply_grads_with_momentum_method
        elif self.optimizer == 'SGD':
            self.apply_grads = self.apply_grads_with_sgd
        elif self.optimizer == 'Adam':
            self.apply_grads = self.apply_grads_with_adam
        else:
            exit("Optimizer not found!")

        self.class_column = class_column
        self.class_values = class_values

    def reset(self):
        self.init_activations()
        self.init_deltas()
        self.init_grads()
        self.init_weights()
        if self.optimizer == 'Momentum':
            self.init_z_directions()
        elif self.optimizer == 'Adam':
            self.init_adam()
    
    def predict(self, instance):
        instance = instance.drop(labels=[self.class_column]).values
        instance = np.array(instance, dtype='float64')
        activations = list(self.propagate(instance))
        max_index = activations.index(max(activations))
        return self.class_values[max_index]
    
    def train(self, x, y, x_test, y_test):
        train_loss, test_loss = [], []  
        n = len(x)
        batch_size = self.batch_size if self.batch_size is not None else n
        num_batches = n // batch_size
        batches_x = np.array_split(x, num_batches)
        batches_y = np.array_split(y, num_batches)
        epochs = tqdm(range(self.epochs))
        self.reset()
        for e in epochs:
            epoch_loss = 0.0
            for batch in range(num_batches):
                sum_loss = 0.0
                self.reset_grads()
                true_batch_size = len(batches_x[batch])

                for i in range(true_batch_size):
                    xi, yi = batches_x[batch][i], batches_y[batch][i].reshape(-1,1)
                    fx = self.propagate(xi)
                    sum_loss += self.cost(fx, yi)
                    self.backpropagate(fx, yi)

                regularized_loss = sum_loss/true_batch_size + self.regularization_cost(true_batch_size)
                epoch_loss += sum_loss
                self.add_regularization_to_grads(true_batch_size)
                self.apply_grads()

            testloss = 0.0
            for i in range(len(x_test)):
                xi, yi = x_test[i], y_test[i].reshape(-1,1)
                fx = self.propagate(xi)
                testloss += self.cost(fx, yi)
            test_loss.append(testloss/len(x_test) + self.regularization_cost(len(x_test)))
            train_loss.append(epoch_loss/n + self.regularization_cost(n))

            epochs.set_description('Epoch {}: train loss = {:.5f} test loss = {:.5f}'.format(e+1, train_loss[-1], test_loss[-1]))

        return train_loss, test_loss

    def train_numerically(self, x, y):
        n = len(x)
        self.reset()
        for e in range(self.epochs):
            print('\nEpoch {}:'.format(e+1))
            self.reset_grads()
            print('Gradientes numericos:')
            for i in range(n):
                xi, yi = x[i], y[i].reshape(-1,1)
                self.calculate_numerical_gradients(xi, yi)
            self.add_regularization_to_grads(n)
            print(self.gradients_as_strings())
            numerical_grads = copy.deepcopy(self.grads)
            self.reset_grads()
            print('\nGradientes backpropagation:')
            for i in range(n):
                xi, yi = x[i], y[i].reshape(-1,1)
                fx = self.propagate(xi)
                self.backpropagate(fx, yi)
            self.add_regularization_to_grads(n)
            print(self.gradients_as_strings(), '\n')
            self.print_grad_diff(numerical_grads, self.grads)
            self.apply_grads()

    def print_grad_diff(self, numerical, backpropagation):
        for theta in range(len(numerical)):
            mean_diff = np.mean(np.abs(numerical[theta] - backpropagation[theta]))
            print('Erro entre grandiente via backprop e grandiente numerico para Theta%d: %.10f' %(theta+1, mean_diff))

    def calculate_numerical_gradients(self, x, y):
        epsilon = 1e-8
        for layer in range(len(self.weights)):
            i, dims = self.weights[layer].shape
            for neuron in range(dims):
                for next_layer in range(i):
                    # Cost summing epsilon
                    self.weights[layer][next_layer, neuron] += epsilon
                    plus_epsilon_propagation = self.propagate(x)
                    plus_epsilon_cost = self.cost(plus_epsilon_propagation, y)
                    # Cost subtracting epsilon
                    self.weights[layer][next_layer, neuron] -= (2 * epsilon)
                    less_epsilon_propagation = self.propagate(x)
                    less_epsilon_cost = self.cost(less_epsilon_propagation, y)
                    # Correct weight to initial value
                    self.weights[layer][next_layer, neuron] += epsilon
                    # Calculate gradient
                    gradient = (plus_epsilon_cost - less_epsilon_cost) / (2 * epsilon)
                    # Update gradient
                    self.grads[layer][next_layer][neuron] += gradient


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
            uncertainty = np.multiply(self.activations[layer][1:], (1 - self.activations[layer][1:]))
            np.multiply(np.dot(weights, self.deltas[layer]), uncertainty, out=self.deltas[layer-1])

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

    def apply_grads_with_sgd(self):
        self.weights -= self.alpha * self.grads

    def apply_grads_with_momentum_method(self):
        self.z_directions *= self.beta
        self.z_directions += (1 - self.beta) * self.grads
        self.weights -= self.alpha * self.z_directions
    
    def apply_grads_with_adam(self):
        self.t += 1
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8

        self.m_t = beta_1 * self.m_t + (1-beta_1) * self.grads	      # updates the moving averages of the gradient
        self.v_t = beta_2 * self.v_t + (1-beta_2) * (self.grads**2)	  # updates the moving averages of the squared gradient

        m_cap = self.m_t/(1-(beta_1**self.t))		# calculates the bias-corrected estimates
        v_cap = self.v_t/(1-(beta_2**self.t))		# calculates the bias-corrected estimates
        sqrt_v = np.array([np.sqrt(x) for x in v_cap])
        self.weights -= (self.alpha*m_cap) / (sqrt_v + epsilon)

    def init_adam(self):
        self.t = 0
        self.m_t = np.array([np.zeros((self.architecture[layer+1], self.architecture[layer]+1), dtype='float') for layer in range(self.num_layers-1)])
        self.v_t = np.array([np.zeros((self.architecture[layer+1], self.architecture[layer]+1), dtype='float') for layer in range(self.num_layers-1)])

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
        self.weights = np.array([np.random.normal(size=(self.architecture[layer+1], self.architecture[layer]+1)) for layer in range(self.num_layers-1)])

    def read_weights_from_file(self, file):
        w = []
        with open(file, 'r') as f:
            for line in f:
                w.append([float(x) for x in line.replace(';',',').split(',')])
        self.weights = np.array([np.asmatrix(w[layer]).reshape(self.architecture[layer+1], self.architecture[layer]+1) for layer in range(self.num_layers-1)])

    def init_deltas(self):
        self.deltas = np.array([np.empty((self.architecture[n], 1)) for n in range(1, self.num_layers)])

    def init_z_directions(self):
        self.z_directions = np.array([np.zeros(layer.shape) for layer in self.weights])

    def init_grads(self):
        self.grads = np.array([np.zeros((self.architecture[layer+1], self.architecture[layer]+1)) for layer in range(self.num_layers-1)])

    def build_architecture_from_file(self, architecture):
        with open(architecture, 'r') as f:
            fileread = f.read().splitlines()
        self.regularization_factor = float(fileread[0])
        self.architecture = [int (x) for x in fileread[1:]]

    def gradients_as_strings(self):
        return "\n".join(
            ["; ".join(
                [", ".join(["{:.5f}".format(w) for w in neuron])
                 for neuron in layer]
            ) for layer in self.grads]
        )
