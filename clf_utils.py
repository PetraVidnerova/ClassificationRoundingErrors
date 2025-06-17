import torch
import torch.nn as nn
from preprocessing import stack_linear_layers

def difference_layer(k, inputs=10):

    layer = nn.Linear(inputs, inputs-1).double()

    i = 0
    for j in range(inputs):
        if j == k:
            continue

        for n in range(inputs):
            layer.weight.data[i, n] = 0.0
        layer.bias.data[i] = 0.0

        layer.weight.data[i, k] = -1.0
        layer.weight.data[i, j] = 1.0

        i+= 1

    return layer

def minus_layer(inputs=10):

    layer = nn.Linear(inputs, 2*inputs).double()

    layer.weight.data.fill_(0.0)
    layer.bias.data.fill_(0.0)

    for i in range(inputs):
        layer.weight.data[i, i] = 1.0
        layer.weight.data[inputs+i, i] = -1.0

    return layer

def difference_layer_fixed(k, c, inputs=10, p=0.9):
    # expects 2*inputs inputs from the minus layer

    layer = nn.Linear(2*inputs, inputs).double() 
    layer.weight.data.fill_(0.0)
    layer.bias.data.fill_(0.0)
    
    for j in range(inputs):
        layer.weight.data[j, j] = 1.0
        layer.weight.data[j, inputs+j] = -1.0
        layer.weight.data[j, c] = -p
        layer.weight.data[j, inputs+c] = 1.0
        layer.weight.data[j, inputs+k] = (1-p)
        
        
    return layer

def fix_me_layer(inputs=10):
    layer = nn.Linear(inputs, inputs).double()

    layer.weight.data.fill_(0.0)
    ##layer.bias.data.fill_(-2.0)
    layer.bias.data.fill_(-0.0)

    for j in range(inputs):
        layer.weight.data[j, j] = 1.0

    return layer
    

def concat_lin_layers(layers):
    
    inputs = [l.weight.data.shape[1] for l in layers]
    all_inputs = sum(inputs)
    
    wide_lin_layer_weights = []

    wide_layer_weight = torch.hstack([layers[0].weight.data,
                                      torch.zeros((layers[0].weight.data.shape[0], all_inputs-inputs[0]), dtype=torch.float64)])
    wide_lin_layer_weights.append(wide_layer_weight)

    for i, layer in enumerate(layers):
        if i == 0 or i == len(layers)-1:
            continue
        wide_layer_weight = torch.hstack([torch.zeros((layers[i].weight.data.shape[0], sum(inputs[:i])), dtype=torch.float64),
                                          layers[i].weight.data,
                                          torch.zeros((layers[i].weight.data.shape[0], sum(inputs[i+1:])), dtype=torch.float64)])
        wide_lin_layer_weights.append(wide_layer_weight)

    wide_layer_weight = torch.hstack([torch.zeros((layers[-1].weight.data.shape[0], all_inputs-inputs[-1]), dtype=torch.float64),
                                      layers[-1].weight.data])
    wide_lin_layer_weights.append(wide_layer_weight)
        

    wide_weight = torch.vstack(wide_lin_layer_weights)
    layer = nn.Linear(wide_weight.shape[1], wide_weight.shape[0])
    layer.weight.data = wide_weight
    layer.bias.data = torch.hstack([l.bias.data for l in layers])

    return layer
    

def exp_layer(inputs=10):

    lin_layers = []
    output_layers = []
    
    for i in range(inputs):
        net = torch.load("exp_network_10.pt")
        for k, layer in enumerate(net):
            if k == 0:
                lin_layers.append(layer)
            elif k == 2:
                output_layers.append(layer)


    lin_layer = concat_lin_layers(lin_layers)
    output_layer = concat_lin_layers(output_layers)
    
    exp_layers = [
        lin_layer,
        nn.ReLU(),
        output_layer
    ]

    return exp_layers

def sum_layer(inputs):
    layer = nn.Linear(inputs, 1).double()
    layer.weight.data.fill_(1.0)
    layer.bias.data[0] = 0

    return layer


if __name__ == "__main__":

    inputs = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float64)
    inputs = inputs.reshape(1, 10)
    
    outputs = difference_layer(2)(inputs)

    print(inputs)
    print(outputs)


    outputs = sum_layer(9)(outputs)
    print(outputs)
    

    exp = nn.Sequential(*exp_layer(inputs=10))
    print(exp)

    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


    net = minus_layer(10)
    inputs = torch.tensor(x, dtype=torch.float64)
    out = net(inputs)
    print(out)
    exit()
    

    
    outputs = exp(inputs)
    print(outputs)

    single_exp = torch.load("exp_network3.pt")
    
    import numpy as np
    for xx, output in zip(x, outputs):
        inp = torch.tensor([xx], dtype=torch.float64).resize(1,1)
        print(np.exp(xx), output, single_exp(inp))
    
