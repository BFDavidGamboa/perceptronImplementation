import numpy as np

output = 0


def print_inputs(print_input, print_weights, print_bias):
    print("array of inputs", print_input)
    print("array of weights", print_weights)
    print("array of bias", print_bias)


def sum_neuron_inputs_print(array_input, array_weight, bias_value, output):
    for input in array_input:
        index_input = inputs.index(input)
        output += input * (array_weight[index_input])
        print_inputs(array_input, array_weight, bias_value)


inputs = [1, 2, 3]
weights = [0.0, 0.0, 0.0]
# To train a neuron we'll be using weights
# initialized as 0 in order to start
# then these values are going to change

weights = [0.2, 0.8, -0.5]
# we setup a random value per neuron
bias = 2

sum_neuron_inputs_print(inputs, weights, bias, output)
print_inputs(inputs, weights, bias)


print("Sum for each neural input for a single 3 input neuron ", output)

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

# we setup a random value per neuron
print_inputs(inputs, weights, weights)
print(output + bias)
