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
        pass
    
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        pass
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
        #avg loss
        data_loss = np.mean(sample_losses)
        #return loss
        return data_loss
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
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
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



#create dataset
X,y = spiral_data(samples=100, classes = 3)

#dense layer with 2 input and 3 output
dense1 = Layer_Dense(2,3)

#create ReLU activation layer to be used with the Dense Layer
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (output of last layer) and 3 output values
dense2 = Layer_Dense(3,3)

#SoftMax Activation
activation2 = Activation_Softmax()

#create loss function
loss_function = Loss_CategoricalCrossentropy()

#forward pass of dataset
dense1.forward(X)

#forward pass through activation function
activation1.forward(dense1.output)

#output of activation is input of dense2
dense2.forward(activation1.output)

#activation function pass with output of dense2
activation2.forward(dense2.output)

print(activation2.output[:5])

#output of second dense layer and return loss
loss = loss_function.calculate(activation2.output, y)

print('loss:', loss)

predictions = np.argmax(activation2.output, axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis = 1)
accuracy = np.mean(predictions == y)
print("acc: ", accuracy)    