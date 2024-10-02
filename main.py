import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        #initialize layer weights and biases
    
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        
    def backward(self,dvalues):
        #gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases=np.sum(dvalues,axis=0,keepdims=True)
        
        #gradient on values
        self.dinputs = np.dot(dvalues,self.weights.T)

#ReLU_Activation
class Activation_ReLU:
    
    #forward pass
    def forward(self, inputs):
        self.inputs = inputs
        #calculate output values from inputs
        self.output = np.maximum(0, inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
#softmaxActivation
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        #get un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))

        #normalize them
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
    def backward(self, dvalues):
        #create uninitialized array
        self.dinputs = np.empty_like(dvalues) #empty like makes the array have same dimension as dvalues
        
        #enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #flatten output array
            single_output = single_output.reshape(-1, 1)
            #calculate Jacobian matrix of this output, In vector calculus, the Jacobian matrix of a vector-valued function of several variables is the matrix of all its first-order partial derivatives.
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #calculate sample-wise gradient
            #add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    #calculate loss given output and truth values
    def calculate(self, output, y):
        #sample losses
        sample_losses = self.forward(output, y)
        #return avg loss
        return np.mean(sample_losses)
#Loss_CategoricalCrossentropy
class Loss_CategoricalCrossentropy(Loss):
    
    
    def forward(self, y_pred, y_true):
        #find num samples per batch
        samples = len(y_pred)
        #clip data to prevent div by 0
        #clip both sides to not drag mean to any end
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        #prob for target values, only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        #mask values only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        #losses
        #negative_log_likelihoods = -np.log(correct_confidences)
        return -np.log(correct_confidences)
    
    def backward(self,dvalues,y_true):
        
        samples = len(dvalues) #sample size
        labels = len(dvalues[0]) #labels per sample, use this as a way to count them
        
        #if labels are sparse turn them into a one hot vector (a method of representing characters or words by a vector where only one element is set to one and all others are zero, based on their position in the vocabulary)
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true] #np.eye makes a RRE form for the matrix
        
        #calculate gradients
        self.dinputs = -y_true / dvalues
        
        #normalize gradient
        self.dinputs = self.dinputs / samples
#Softmax Classifier - combined Softmax activation
# and cross-entropy loss for faster backward step

class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    #create objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    def forward(self, inputs, y_true):
        #Output layer's activation function
        self.activation.forward(inputs)
        #set the output
        self.output = self.activation.output
        #calculate and return loss
        return self.loss.calculate(self.output, y_true)
    def backward(self,dvalues,y_true):
        #num samples
        samples = len(dvalues)
        
        #if one-hot encoded, make them discrete
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        
        self.dinputs = dvalues.copy()
        #calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        #normalize
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    #initialize optimizer with set settings
    def __init__(self, learning_rate = 1., decay = 0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate 
        self.decay = decay 
        self.iterations = 0
        self.momentum = momentum
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        if self.momentum:
            #if layer does not have momentum arrays, make them
            if not hasattr(layer, "weight_momentums"):
                #if no momentum for weights, none for bias either
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
                # Build weight updates with momentum - take previous 
                # updates multiplied by retain factor and update with 
                # current gradients
            weight_updates= self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights 
            layer.weight_momentums = weight_updates


            #build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates= -self.current_learning_rate * layer.dweights 
            bias_updates= -self.current_learning_rate * layer.dbiases
            
        layer.weights += weight_updates 
        layer.biases += bias_updates
        
        
class Optimizer_AdGrad:

    #initialize optimizer with set settings
    def __init__(self, learning_rate = 1., decay = 0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate 
        self.decay = decay 
        self.iterations = 0
        self.epsilon = epsilon
        
    def post_update_params(self):
        self.iterations += 1
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
            
    def update_params(self, layer):
        # if layer does not have cache arrays, make them
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2 
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization 
        # with square-rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


class Optimizer_RMSprop:
    #initialize optimizer with set settings
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate 
        self.decay = decay 
        self.iterations = 0
        self.rho = rho
        self.epsilon = epsilon
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            # Initialize cache for weights and biases if not already present
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with current squared gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

            
    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1)

        # Correct momentum with bias correction
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square-rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        
#create dataset
X, y = spiral_data(samples=100,classes=3)

#dense layer with 2 input and 3 output
dense1 = Layer_Dense(2,64)

#create ReLU activation layer to be used with the Dense Layer
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (output of last layer) and 3 output values
dense2 = Layer_Dense(64,3)

#create soft max classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate = 0.05,decay = 5e-7)

for epoch in range(10001):

    #forward pass of dataset
    dense1.forward(X)

    #forward pass through activation function
    activation1.forward(dense1.output)

    #output of activation is input of dense2
    dense2.forward(activation1.output)

    #takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)

    #calculate accuracy from output of act2 and targets, calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis = 1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch:{epoch}, '+
              f'acc:{accuracy:.3f}, '+
              f'loss:{loss:.3f}, '+
              f'lr:{optimizer.current_learning_rate}')
        2
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params() 
    optimizer.update_params(dense1) 
    optimizer.update_params(dense2) 
    optimizer.post_update_params()


#notes
#input to loss function , y-hat is output of Activation function S_ij
#Normalizing is a process used to scale or adjust data so that it falls within a certain range or follows a specific distribution. 
#Each full pass through all of the training data is called anepoch