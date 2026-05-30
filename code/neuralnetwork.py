import numpy as np
#single neuron with 4 inputs, weights and a bias
'''
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
print(output)
'''
#layer of neurons

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
#this is the same as *
outputs = np.dot(weights, inputs) + biases 
print(outputs)
#*
output = [
    float(np.dot(inputs, weights[0]) + bias1),
    float(np.dot(inputs, weights[1]) + bias2),
    float(np.dot(inputs, weights[2]) + bias3)
]
#*
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = float(np.dot(inputs, neuron_weights) + neuron_bias)
    layer_outputs.append(neuron_output)
print(output)
print(layer_outputs)
