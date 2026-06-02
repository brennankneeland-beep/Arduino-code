
#this will eventually take an input from a atmospheric 
#pressure guage and output a prediction of the weather 
#for the next day to be displayed on a LCD screen.

import numpy as np


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
X, y = spiral_data(100, 3)
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
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output =[]
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
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

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
#print(layer1.output)
print(activation1.output)