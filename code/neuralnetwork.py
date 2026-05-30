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
bias1 = 2
bias2 = 3
bias3 = 0.5
output = [
    float(np.dot(inputs, weight1) + bias1),
    float(np.dot(inputs, weight2) + bias2),
    float(np.dot(inputs, weight3) + bias3)
]

print(output)
