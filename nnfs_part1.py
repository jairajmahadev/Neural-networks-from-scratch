# using dot product to calculate output of neurons
# output is calculated as input * weight + bias
# this example calculates outputs of three neurons is a layer using dot product

import numpy as np

inputs = [1,2,3,2.5]
weights = [[0.2,0.8, -0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

output = np.dot(weights,inputs) + biases
print(output)

# passing inputs as a batch
inputs = [[1,2,3,2.5],
          [2,5,-1,2],
          [-1.5,2.7,3.3,-0.8]]

# now weights matrix needs to be transposed for dot fun to work
output = np.dot(np.array(inputs),np.array(weights).T) + biases
print(output)