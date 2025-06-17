import time
import click
import numpy as np
import torch
import tqdm

from preprocessing import create_comparing_network, eval_one_sample
from preprocessing import create_comparing_network_classifier

from network import load_network, SmallConvNet, SmallDenseNet
from dataset import create_dataset
from linear_utils import create_c, create_upper_bounds, optimize, create_p_bounds, create_p2_bounds
from linear_utils import TOL, TOL2

from quant_utils import lower_precision
from clf_utils import difference_layer, exp_layer, sum_layer, minus_layer, difference_layer_fixed, fix_me_layer

CLASS = 1
PROBA = 0.7

def check_upper_bounds(A, b, input1, input2):

    A = A.cpu()

    if input1 is not None:
        input1 = input1.cpu()
        input1 = torch.hstack([torch.tensor(1),
                               input1.reshape(-1)])

    if input2 is not None:
        input2 = torch.hstack([torch.tensor(1),
                               torch.tensor(input2)])

    if input1 is not None:
        result = A @ input1
        print("Check upper bounds 1: ", torch.all(result <= TOL + TOL2))
        if not torch.all(result <= TOL + TOL2):
            print("upper bounds 1 failed")
    
        assert torch.all(result <= TOL + TOL2)

    if input2 is not None:
        result = A @ input2 
        print("Check upper bounds 2: ", torch.all(result <= TOL + TOL2 ))
        if not torch.all(result <= TOL + TOL2):
            print("upper bounds 2 failed")
        assert torch.all(result <= TOL + TOL2)
    
        #   wrong_indexes = torch.logical_not(result <= TOL + TOL2)
        #   print(wrong_indexes.sum())
    
        #   print(result[wrong_indexes])

def check_saturations(net, input1, input2):
    
    input2 = torch.tensor(input2).reshape(1, 1, 28, 28).cuda()

    saturation1 = eval_one_sample(net, input1)
    saturation2 = eval_one_sample(net, input2)

    saturation1 = torch.hstack(saturation1)
    saturation2 = torch.hstack(saturation2)
    
    print("Check saturations", torch.all(saturation1 == saturation2).item())
    assert torch.all(saturation1 == saturation2)

def net_for_condition(c):

    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net2 = load_network(MODEL, NETWORK)
    net2 = next(iter(net2.children())) # get it as plain Sequential
    
    
    diff_layer = difference_layer(c)
    net2.add_module("diff_layer", diff_layer)

        
    wide_exp_layers = exp_layer(inputs=9)
    
    for i, layer in enumerate(wide_exp_layers):
        net2.add_module(f"exp_{i}", layer)

    sum_layer2 = sum_layer(10-1) # FIXME instead of 10 number of classes
    net2.add_module(f"sum", sum_layer2)

    return net2

def classify(net, x):
    scores = net(x)
    _, predictions = scores.max(1)
    return predictions[0].item()

def loss(net, net2, x):

    scores1 = net(x)
    scores2 = net2(x)

    probs1 = torch.nn.functional.softmax(scores1)
    probs2 = torch.nn.functional.softmax(scores2)

    # print("Orig net probs:   ", probs1)
    # print("Rounded net probs:", probs2)

    probs2 = torch.log(probs2)

    return -(probs1 * probs2).sum()
    
    

def test_classification(x):
    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK)
    net2 = load_network(MODEL, NETWORK)
    net2 = lower_precision(net2, bits=4)

    return classify(net, x), classify(net2, x)
          
    



def net_to_optim(k, rounding=True, no_exp_sum=False, fixme=True):
    
    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

#    net = load_network(MODEL, NETWORK)
    net2 = load_network(MODEL, NETWORK)

    if rounding:
        net2 = lower_precision(net2, bits=4)
    net2 = next(iter(net2.children())) # get it as plain Sequential
    
    
    # diff_layer = difference_layer(k)
    # net2.add_module("diff_layer", diff_layer)

    minus_layer1 = minus_layer(inputs=10)
    net2.add_module("minus_layer", minus_layer1)

    net2.add_module("ReLU", torch.nn.ReLU())

    diff_layer = difference_layer_fixed(k, CLASS, inputs=10, p=PROBA)
    net2.add_module("diff_layer", diff_layer)
    
    if not no_exp_sum:
        if fixme:
            fx = fix_me_layer(inputs=10)
            net2.add_module("fix_me", fx)
        
        wide_exp_layers = exp_layer()

        for i, layer in enumerate(wide_exp_layers):
            net2.add_module(f"exp_{i}", layer)

        sum_layer2 = sum_layer(10) # FIXME instead of 10 number of classes
        net2.add_module(f"sum", sum_layer2)

    return net2

def maximize_k(inputs, k):

    target_net = net_to_optim(k).cuda()
    

    # outputs = target_net(inputs)
    # print(outputs)


    # net = load_network(SmallDenseNet, "mnist_dense_net.pt")
    # net = lower_precision(net, bits=4)
    # outputs = net(inputs)

    # diffs = torch.tensor([(outputs[0, i] - outputs[0, k]) for i in range(10) if i!=k], dtype=torch.float64)

    # e = torch.exp(diffs)
    # print(e)

    # print(torch.sum(e))
    
    # exit()

    # min c @ x -> -1 because we are maximizing
    c = -1 * create_c(target_net, inputs)

    # upper bounds - bounds given by saturations
    A_ub, b_ub = create_upper_bounds(target_net, inputs)

    # bound for  <= (1-p)/p from eq (16)
    orig_net = net_for_condition(CLASS).cuda()
    A_ub2, b_ub2 = create_upper_bounds(orig_net, inputs)

    A_ub3, b_ub3 = create_p_bounds(orig_net, inputs, PROBA)

    A_ub4, b_ub4 = create_p2_bounds(net_to_optim(k, no_exp_sum=True).cuda(),
                                    inputs,
                                    PROBA)
    # print(A_ub3.shape)
    # print(b_ub3.shape)
    
    
    A_ub = torch.vstack([A_ub, A_ub2, A_ub3, A_ub4])
    b_ub = torch.hstack([b_ub, b_ub2, b_ub3, b_ub4])

    # A_ub = torch.vstack([A_ub, A_ub4])
    # b_ub = torch.hstack([b_ub, b_ub4])


    N = len(inputs.flatten())
    # A_eq @ x == b_eq
    # bias input == 1
    A_eq = torch.zeros((1, N+1)).double()
    A_eq[0, 0] = 1.0
    b_eq = torch.zeros((1,)).double()
    b_eq[0] = 1.0                    
            
    # l <= x <= u 
    l = -0.5
    u = 3.0

    res = optimize(c, A_ub, b_ub, A_eq, b_eq, l, u)
    
    if res.success:
                
        err, x = res.fun, res.x
        #print("result:", -err)
        
        assert np.isclose(x[0], 1.0)
            
        y = torch.tensor(x[1:], dtype=torch.float64).reshape(1, -1).cuda()
        return -err, y

    else:
        #print("Solution not found.")
        return None, None
    
@click.command()
@click.argument("start", type=int)
@click.argument("end", type=int)
@click.option("-b", "--bits", default=16)
@click.option("--outputdir", default="results_exp")
def main(start, end, bits, outputdir):

    BATCH_SIZE = 1
    
    data = create_dataset(train=False, batch_size=BATCH_SIZE)
    count = 0

    NETWORK="mnist_dense_net.pt"
    MODEL = SmallDenseNet 
    LAYERS = 4
    INPUT_SIZE = (1, 28, 28) 
    N = 1 * 28 * 28 

    net = load_network(MODEL, NETWORK)
    net2 = load_network(MODEL, NETWORK)
    net2 = lower_precision(net2, bits=4)
    
    for i, (inputs, labels) in tqdm.tqdm(enumerate(data), total=10000):
        if i < start or i >= end:
            continue
        label = labels[0].item()
        inputs = inputs.cuda().double()
        
        if label != CLASS:
            continue

        if classify(net, inputs) != CLASS:
            continue

        count += 1
        

        maximas = []
        points = []
        for k in range(10): # todo FIX number 10 to number classes
            if k == CLASS:
                continue
            mx, value = maximize_k(inputs, k)
            maximas.append(mx)
            points.append(value)
            
        
        with open(f"{outputdir}/results_{start}_{end}_8bits.csv", "a") as f:
            values = [val for val in maximas if val is not None]
            if values:
                max_value = max(values) #f"{max(values):.6f}"
                x = points[np.argmax(values)]
                #                print(x)
                
                torch.save(x, f"{outputdir}/x{i}.pt")

                x_loss = loss(net, net2, inputs)
                max_x_loss = loss(net, net2, x)
                est_loss = max_value #np.log(max_value)#+2
                
                orig_class, rounded_class = test_classification(x)
                
                print(f"{i},{orig_class},{rounded_class},{x_loss},{max_x_loss},{est_loss}")
                # import matplotlib.pyplot as plt
                # plt.imshow(x.cpu().numpy().reshape(28,28))
                # plt.savefig("x.png")
                # exit()                
            else:
                max_value = None
            str_max = f"{(np.log(max_value)+10):.6f}" if max_value is not None else "None"
            print(f"{i},{str_max},{maximas}", file=f)

        

        
    print(count)
        
        
        
if __name__ == "__main__":

    # test_squeeze() # 1.
    #test_compnet() # 2.
    #test_squeezed_compnet() # 3.


    main()
