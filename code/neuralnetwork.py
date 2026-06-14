
#this will eventually take an input from a atmospheric 
#pressure guage and output a prediction of the weather 
#for the next day to be displayed on a LCD screen.

import numpy as np
import math

#single neuron with 4 inputs, weights and a bias
'''
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
print(output)
'''
#layer of neurons with 4 inputs and 3 neurons, 
#each neuron has its own weights and bias
'''
inputs = [1, 2, 3, 2.5]
weight1  = [0.2, 0.8, -0.5, 1.0]
weight2  = [0.5, -0.91, 0.26, -0.5]
weight3  = [-0.26, -0.27, 0.17, 0.87]
weights = [weight1, weight2, weight3]
bias1 = 2
bias2 = 3
bias3 = 0.5
biases = [bias1, bias2, bias3]
output = np.dot(weight1, inputs) + bias1
print(output)
#this is the same as any * 
#this one is the most efficient, however
outputs = np.dot(weights, inputs) + biases 
print(outputs)
#*
'''
'''
output = [
    float(np.dot(inputs, weights[0]) + bias1),
    float(np.dot(inputs, weights[1]) + bias2),
    float(np.dot(inputs, weights[2]) + bias3)
]
'''
#*
'''
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = float(np.dot(inputs, neuron_weights) + neuron_bias)
    layer_outputs.append(neuron_output)
print(output)
print(layer_outputs)
'''

#batch size of 3, 4 inputs and 3 neurons, two layers of neurons

'''
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]
output = np.dot(inputs, np.array(weights).T) + biases
#print(output)
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
layer1_output = np.dot(inputs, np.array(weights).T) + biases
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
print(layer2_output)
'''
#make it an object
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#X, y = spiral_data(100, 3)
'''
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
'''
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

'''
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)
layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
'''
#rectufied linear activation function
'''
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output =[]
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
'''
#same thing but with max function
'''
output = []
for i in inputs:
    output.append(max(0, i))
print(output)
'''
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
'''
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
print(layer1.output)
print(activation1.output)
'''
#softmax activation function
'''
layer_outputs = [4.8, 1.21, 2.385]
E = math.e
exp_values = []
exp_values = np.exp(layer_outputs)
print(exp_values)
norm_values = exp_values / np.sum(exp_values)
print(norm_values)
print(sum(norm_values))
'''
'''
this is the same as above but with a for loop instead of numpy
for i in layer_outputs:
    exp_values.append(E**i)
print(exp_values)
norm_base = sum(exp_values)
normalized_values = []
for i in exp_values:
    normalized_values.append(i/norm_base)
print(normalized_values)
print(sum(normalized_values))
'''
'''
layer_outputs = [[4.8, 1.21, 2.385], 
                 [8.9, -1.81, 0.2], 
                 [1.41, 1.051, 0.026]]
exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) #axis 1 is the row, axis 0 is the column
print(norm_values)
print(np.sum(norm_values, axis=1))#keepdins true keeps the dimensions of the array the same, so it can be used for broadcasting
'''
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1 , keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
'''
X, y = spiral_data(samples = 100,classes = 3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output)
'''
#calculating loss
'''
softmax_output_example = [0.7, 0.1, 0.2]
target_output_example = [1, 0, 0]
target_class = 0#this is the index of the correct class in the target output
loss = -(math.log(softmax_output_example[target_class])*target_output_example[target_class]+
         math.log(softmax_output_example[1])*target_output_example[1]+
         math.log(softmax_output_example[2])*target_output_example[2])
print(loss)
#same as 
loss = -math.log(softmax_output_example[target_class])
print(loss)
'''
'''
softmax_output_examples = np.array([[0.7, 0.1, 0.2],
                                    [0.5, 0.1, 0.4],
                                    [0.02, 0.9, 0.08]])
class_Targets = [0, 1, 1]
print(softmax_output_examples[[0, 1, 2], class_Targets])
print(softmax_output_examples[range(len(softmax_output_examples)), class_Targets])
loss = -np.log(softmax_output_examples[range(len(softmax_output_examples)), class_Targets])
print(loss)
average_loss = np.mean(loss)
'''
class loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
class Loss_CategoricalCrossentropy(loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])#note 0 can be replaced with any number as long as its not out of bounds
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
'''
predictions = np.argmax(softmax_output_examples, axis=1)
accuracy = np.mean(predictions == class_Targets)
print('Accuracy:', accuracy)
'''
#back propogation for a singe neuron (minimize output)
'''
x = [1, -2, 3]#inputs
w = [-3, -1, 2]#weights
b = 1#bias
for i in range(600):
    xw0 = x[0]*w[0]
    xw1 = x[1]*w[1]
    xw2 = x[2]*w[2]
    z = xw0 + xw1 + xw2 + b
    #print(xw0, xw1, xw2)
    #print(z)
    y = max(0, z)
    #print(y)
    dval = 1
    drelu_dz = dval * (1. if z > 0 else 0.)
    #print(drelu_dz)
    dsum_dxw0 = 1
    drelu_dxw0 = drelu_dz * dsum_dxw0

    dsum_dxw1 = 1
    drelu_dxw1 = drelu_dz * dsum_dxw1

    dsum_dxw2 = 1 
    drelu_dxw2 = drelu_dz * dsum_dxw2

    dsum_db = 1
    drelu_db = drelu_dz * dsum_db
    #print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)
    dmul_dx0 = w[0]
    drelu_dx0 = drelu_dxw0 * dmul_dx0
    dmul_dx1 = w[1]
    drelu_dx1 = drelu_dxw1 * dmul_dx1
    dmul_dx2 = w[2]
    drelu_dx2 = drelu_dxw2 * dmul_dx2

    dmul_dw0 = x[0]
    drelu_dw0 = drelu_dxw0 * dmul_dw0
    dmul_dw1 = x[1]
    drelu_dw1 = drelu_dxw1 * dmul_dw1
    dmul_dw2 = x[2]
    drelu_dw2 = drelu_dxw2 * dmul_dw2
    #print(drelu_dx0, drelu_dx1, drelu_dx2)
    #print(drelu_dw0, drelu_dw1, drelu_dw2)
    drelu_dx0 = dval*(1 if z > 0 else 0.)*w[0]
    dx = [drelu_dx0, drelu_dx1, drelu_dx2]
    dw = [drelu_dw0, drelu_dw1, drelu_dw2]
    db = drelu_db
    #print(w, b)
    w[0] -= 0.001*dw[0]
    w[1] -= 0.001*dw[1]
    w[2] -= 0.001*dw[2]
    b -= 0.001*db
    print(y)
print (w, b)
'''
#back propogation for a layer of neurons
'''
dvals = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3]])
inputs = np.array([[1, 2, 3, 2.5],
                   [2.0, 5.0, -1.0, 2.0],
                   [-1.5, 2.7, 3.3, -0.8]])
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T
biases = np.array([[2, 3, 0.5]])
dbiases = np.sum(dvals, axis=0, keepdims=True)
dweights = np.dot(inputs.T, dvals)
print(dbiases)
print(dweights)
'''
'''
dx0 = sum(weights[0]*dvals[0])
dx1 = sum(weights[1]*dvals[0])
dx2 = sum(weights[2]*dvals[0])
dx3 = sum(weights[3]*dvals[0])

dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)
'''
#same as above but with numpy
'''
dinputs = np.dot(dvals, weights.T)
print(dinputs)
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])
dvals = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
'''
'''
drelu = np.zeros_like(z)
drelu[z > 0] = 1
print(drelu)
drelu *= dvals
print(drelu)
'''
'''
drelu = dvals.copy()
drelu[z <= 0] = 0
print(drelu)
'''
#finally:
''' 
dvals = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3]])
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T
biases = np.array([[2, 3, 0.5]])

layer_outputs = np.dot(inputs, weights) + biases


relu_outputs = np.maximum(0, layer_outputs)

drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

weights += -0.001*dweights
biases += -0.001*dbiases
print(weights)
print(biases)
'''
'''
softmax_outputs = [0.7, 0.1, 0.2]
softmax_outputs = np.array(softmax_outputs).reshape(-1, 1)
print(softmax_outputs)
print(np.diagflat(softmax_outputs))
print(np.dot(softmax_outputs, softmax_outputs.T))
x = (np.diagflat(softmax_outputs) - np.dot(softmax_outputs, softmax_outputs.T))
print(np.dot(x, softmax_outputs))
'''
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward()
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backwards(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs/samples
softmax_outputs = np.array([[.7, .1, .3],
                           [.1, .5, .4],
                           [.02, .9, .08]])
class_targets = np.array([0, 1, 1])
softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backwards(softmax_outputs, class_targets)
dvals1 = softmax_loss.dinputs
activation = Activation_Softmax()
lo = Loss_CategoricalCrossentropy()
activation.output = softmax_outputs
lo.backward(softmax_outputs, class_targets)
activation.backward(lo.dinputs)
dvals2 = activation.dinputs
print(dvals1)
print(dvals2)