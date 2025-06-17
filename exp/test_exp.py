import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt

XMIN = -50
XMAX =  16

def piecewise_linear_approx_exp(interval_edges):
    coeficients = []

    for i in range(len(interval_edges) - 1):
        x0 = interval_edges[i]
        x1 = interval_edges[i + 1]
        y0 = np.exp(x0)
        y1 = np.exp(x1)
        
        a = (y1 - y0) / (x1 - x0)
        b = (x1 * y0 - x0 * y1 ) / (x1 - x0)
 
        coeficients.append((a, b))

    r = len(coeficients)
    assert r == len(interval_edges) - 1 
    
    coeficients = [(0, np.exp(interval_edges[0]))] + coeficients
    interval_edges = np.concatenate([np.array([XMIN]), interval_edges])
    r += 1

    print(interval_edges)
    print(coeficients)


    # create coresponing neural network
    hidden_layer = nn.Linear(1, 2*r-1).double()
    output_layer = nn.Linear(2*r-1, 1).double()

    interval_edges_star = interval_edges
    
    # hidden layer 
    y = np.empty(r)
    for i in range(0, r):
        print(i)
        print(interval_edges_star[i])
        y[i] = np.exp(interval_edges_star[i])
    #y[0] = 0

    for i in range(r):
        m, b = coeficients[i]
        hidden_layer.weight.data[i, 0] = m
        hidden_layer.bias.data[i] = b - y[i]
        if i < r - 1:
            hidden_layer.weight.data[r+i, 0] = m
            hidden_layer.bias.data[r+i] = b - y[i+1]

    # output layer
    for i in range(r):
        output_layer.weight.data[0, i] = +1.0
        if i < r-1:
            output_layer.weight.data[0, r+i] = -1.0
    output_layer.bias.data[0] = 0

    net = nn.Sequential(hidden_layer, nn.ReLU(), output_layer)
    return net


def iterate_over_points(interval_edges, steps=100):
    a = np.array(interval_edges)
    r = len(interval_edges)-2

    for step in range(steps):
        for i in range(1, len(a)-1):
            if step % r == i or (step % r == 0 and i == len(a)-2):
                a[i] = np.log((np.exp(a[i+1]) - np.exp(a[i-1]))/(a[i+1] - a[i-1]))
            else:
                a[i] = a[i]

    return a

if __name__ == "__main__":

    x_range = (-5, 5)

    intervals = np.linspace(*x_range, 16)
    print(intervals)
    
    intervals = iterate_over_points(intervals)
    print(intervals)
    exit()
    
    intervals = np.concatenate([intervals, np.array([XMAX])])

    y_approx = piecewise_linear_approx_exp(intervals)


    torch.save(y_approx, "exp_network_10.pt")

    x = np.linspace(-1, 3, 10000)

    print(y_approx)

    input_x = torch.tensor(x, dtype=torch.float64).reshape(-1, 1)

    y_approx_values = y_approx(input_x).detach().numpy().flatten() 
    y_exact = np.exp(x)

    
    plt.plot(x, y_exact, label='exp(x)', color='blue')
    plt.plot(x, y_approx_values, label='Aproximace exp(x)', color='red')
    plt.title('Aproximace exp(x) pomocí částečně lineární funkce')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show() 
