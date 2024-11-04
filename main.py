import numpy as np
import nnfs
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weights_regularizer_l1=0, weights_regularizer_l2=0,
                 biases_regularizer_l1=0, biases_regularizer_l2=0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weights_regularizer_l1 = weights_regularizer_l1
        self.weights_regularizer_l2 = weights_regularizer_l2
        self.biases_regularizer_l1 = biases_regularizer_l1
        self.biases_regularizer_l2 = biases_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weights_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weights_regularizer_l1 * dL1

        if self.weights_regularizer_l2 > 0:
            self.dweights += 2 * self.weights_regularizer_l2 * self.weights

        if self.biases_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.biases_regularizer_l1 * dL1

        if self.biases_regularizer_l2 > 0:
            self.dbiases += 2 * self.biases_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)



class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.input = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class Activation_ReLU:
    
    def predictions(self, outputs):
        return outputs
    
    def forward(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0

class Activation_Softmax:
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis = 1)
    
    def forward(self, inputs):
        self.input = inputs
        exp_value = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_value / np.sum(exp_value, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Activation_Sigmoid:
    
    def predictions(self, outputs):
        return (outputs > 0.5) * 1
    
    def forward(self, inputs):
        self.input = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

class Activation_Linear:
    
    def predictions(self, outputs):
        return outputs
    
    def forward(self, inputs):
        self.input = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()


class Optimizer_SGD:
    def __init__(self, learning_rate = 1., decay = 0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iter = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iter))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weights_momentums'):
                layer.weights_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)

            weights_updates = self.momentum * layer.weights_momentums - self.current_learning_rate * layer.dweights
            layer.weights_momentums = weights_updates

            biases_updates = self.momentum * layer.biases_momentums - self.current_learning_rate * layer.dbiases
            layer.biases_momentums = biases_updates
        else:
            weights_updates = -self.current_learning_rate * layer.dweights
            biases_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weights_updates
        layer.biases += biases_updates

    def post_update_params(self):
        self.iter += 1


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., eps=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iter = 0
        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iter))

    def update_params(self, layer):
        if not hasattr(layer, 'weights_cache'):
            layer.weights_momentums = np.zeros_like(layer.weights)
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_momentums = np.zeros_like(layer.biases)
            layer.biases_cache = np.zeros_like(layer.biases)

        layer.weights_momentums = self.beta_1 * layer.weights_momentums + (1 - self.beta_1) * layer.dweights
        layer.biases_momentums = self.beta_1 * layer.biases_momentums + (1 - self.beta_1) * layer.dbiases

        weights_momentums_corrected = layer.weights_momentums / (1 - self.beta_1 ** (self.iter + 1))
        biases_momentums_corrected = layer.biases_momentums / (1 - self.beta_1 ** (self.iter + 1))

        layer.weights_cache = self.beta_2 * layer.weights_cache + (1 - self.beta_2) * layer.dweights**2
        layer.biases_cache = self.beta_2 * layer.biases_cache + (1 - self.beta_2) * layer.dbiases**2

        weights_cache_corrected = layer.weights_cache / (1 - self.beta_2 ** (self.iter + 1))
        biases_cache_corrected = layer.biases_cache / (1 - self.beta_2 ** (self.iter + 1))

        layer.weights += -self.current_learning_rate * weights_momentums_corrected / (np.sqrt(weights_cache_corrected) + self.eps)
        layer.biases += -self.current_learning_rate * biases_momentums_corrected / (np.sqrt(biases_cache_corrected) + self.eps)

    def post_update_params(self):
        self.iter += 1


# Common loss class
class Loss:

    # Regularization loss calculation
    def regularization_loss(self):
        # 0 by default
        regularization_loss = 0

        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weights_regularizer_l1 > 0:
                regularization_loss += layer.weights_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weights_regularizer_l2 > 0:
                regularization_loss += layer.weights_regularizer_l2 * \
                    np.sum(layer.weights * layer.weights)

            # L1 regularization - biases
            # only calculate when factor greater than 0
            if layer.biases_regularizer_l1 > 0:
                regularization_loss += layer.biases_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.biases_regularizer_l2 > 0:
                regularization_loss += layer.biases_regularizer_l2 * \
                    np.sum(layer.biases * layer.biases)

        return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()


class Accuracy:
    def calculate(self, predictions, y):
        comparisions = self.compare(predictions, y)
        
        accuracy = np.mean(comparisions)
        return accuracy

class Accuracy_Regression(Accuracy):
    
    def __init__(self):
        self.precision = None

    def init(self, y, reinit = False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
    
# Model class
class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Initialize trainable_layers as an empty list
        self.trainable_layers = []

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
    
    # Set loss and optimizer
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()
        
        # Count all the objects
        layer_count = len(self.layers)
        
        # Iterate the objects
        for i in range(layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            
            # The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # Add layer to trainable_layers if it has weights
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
       
        # Remember trainable layers in the loss object
        self.loss.remember_trainable_layers(self.trainable_layers)


    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Main training loop
        for epoch in range(1, epochs + 1):

            # Perform the forward pass
            output = self.forward(X)

            # Calculate loss
            data_loss, regularization_loss = \
                self.loss.calculate(output, y)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Perform backward pass
            self.backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}), ' +
                    f'lr: {self.optimizer.current_learning_rate}')

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Performs forward pass
    def forward(self, X):
        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X)
        
        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        # "layer" is now the last object from the list,
        # return its output
        return layer.output
    
    def backward(self ,output, y):
    # First call backward method on the loss
    # this will set dinputs property that the last
    # layer will try to access shortly
        self.loss.backward(output, y)    
        
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)   
        

class Layer_Input:
    def forward(self, inputs):
        self.output = inputs
    
    def finalize(self):
        self.input_layer = Layer_Input()
        
        layer_count = len(self.layers)
        
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer[i - 1]
                self.layers[i].next = self.layers[i + 1]
            
            elif i < layer_count - 1:
                self.layers[i].prev = self.input_layer[i - 1]
                self.layers[i].next = self.loss
            
            else: 
                self.layers[i].prev = self.input_layer[i - 1]
                self.layers[i].next = self.loss



X, y = sine_data()

model = Model()

# layers activations
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

# loss function optimizer
model.set(
loss = Loss_MeanSquaredError(),
optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3),
accuracy  = Accuracy_Regression()
)

model.finalize()
model.train(X, y, epochs = 10000, print_every = 100)

