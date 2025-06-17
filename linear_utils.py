import torch
import torch.nn as nn

from preprocessing import eval_one_sample, squeeze_network, prune_network, get_subnetwork

TOL = 0 #1e-8
TOL2 = 1e-9

def create_c(compnet, inputs):
    assert inputs.shape[0] == 1 # one sample in a batch

    #    wide_inputs = torch.hstack([inputs, inputs]) TODO: delete this line

    # reduce and squeeze compnet 
    saturations = eval_one_sample(compnet, inputs)
    target_net = squeeze_network(prune_network(compnet, saturations))

    W = target_net[-1].weight.data
    b = target_net[-1].bias.data

    assert W.shape[0] == 1
    
    c = torch.hstack([b, W.flatten()])

    return c 

def create_upper_bounds(net, inputs):

    # extract the sequential 
    #    net = next(iter(net.children())) NO NEED FOR COMPNET
    assert isinstance(net, nn.Sequential)
    
    saturations = eval_one_sample(net, inputs)


    A_list = []
    bound_list = [] 
    for i, saturation in enumerate(saturations):
        subnet = get_subnetwork(net, i)
        #print(subnet)
        if i == 0:
            target = subnet
        else:
            target = squeeze_network(prune_network(subnet, saturations[:i]))

        W = target[-1].weight.data
        b = target[-1].bias.data

        
        # saturation: True ~ U, False ~ S   
        W_lower = W[torch.logical_not(saturation).flatten()]
        b_lower = b[torch.logical_not(saturation).flatten()].reshape(-1, 1)
        W_higher = W[saturation.flatten()]
        b_higher = b[saturation.flatten()].reshape(-1, 1)

        bound_for_lower = torch.full((W_lower.shape[0],), -TOL, dtype=torch.float64)
        bound_for_higher = torch.full((W_higher.shape[0],), -TOL, dtype=torch.float64)
        
        W = torch.vstack([W_lower, -1*W_higher])
        b = torch.vstack([b_lower, -1*b_higher])
        
        A = torch.hstack([b, W])
        bound = torch.hstack([bound_for_lower, bound_for_higher])
        
        A_list.append(A)
        bound_list.append(bound)

    return torch.vstack(A_list), torch.hstack(bound_list)


def create_p_bounds(net, inputs, p):
    
    saturations = eval_one_sample(net, inputs)
    target_net = squeeze_network(prune_network(net, saturations))

    W = target_net[-1].weight.data
    b = target_net[-1].bias.data

    assert W.shape[0] == 1
    
    c = torch.hstack([b, W.flatten()]).reshape(1, -1)

    q = (1-p)/p
    
    return c, torch.tensor([q], dtype=torch.float64)
    
def create_p2_bounds(net, inputs, p):
    saturations = eval_one_sample(net, inputs)
    target_net = squeeze_network(prune_network(net, saturations))

    W = target_net[-1].weight.data
    b = target_net[-1].bias.data

    # print(W.shape)
    # print(b.shape)
    b = b.reshape(-1, 1)
    
    c = torch.hstack([b, W])#reshape(1, -1)

    # print(c.shape)
    # exit()
    
    q = [20.0] * 10 # a_r+2

    return c, torch.tensor(q, dtype=torch.float64)
    
    
def optimize(c, A_ub, b_ub, A_eq, b_eq, l, u):
    c = c.cpu().numpy()
    A_ub, b_ub = A_ub.cpu().numpy(), b_ub.cpu().numpy()
    A_eq, b_eq = A_eq.cpu().numpy(), b_eq.cpu().numpy()

    
    from scipy.optimize import linprog

    # res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(l, u), method='highs-ipm', options={"presolve": False,
    #                                                                                  "ipm_optimality_tolerance": 1e-12,
    #                                                                                  "primal_feasibility_tolerance": 1e-10,
    #                                                                                  "dual_feasibility_tolerance": 1e-10})

    # res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(l, u), method='interior-point', options={"presolve": False})
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(l, u))
    
    return res

# def poly_volume(A_ub, b_ub, A_eq, b_eq, l, u):
#     """
#     A_ub < b_ub
#     A_eq = b_eq 
#     l < x < u

#     ===> A <= b
#     """

#     matrices = [A_ub.cpu()]
#     left_sides = [b_ub.cpu()] 

#     # A_eq = b_eq --> A_eq <= b_eq & A_eq >= b_eq 
#     matrices.append(A_eq)
#     left_sides.append(b_eq)

#     matrices.append(-A_eq)
#     left_sides.append(-b_eq)
    
#     # l < x
#     A = -torch.diag(torch.ones(A_eq.shape[1]))
#     b = -torch.full((A_eq.shape[1],), fill_value=l)
    
#     matrices.append(A)
#     left_sides.append(b)

#     # x < u 
#     A = torch.diag(torch.ones(A_eq.shape[1]))
#     b = torch.full((A_eq.shape[1],), fill_value=u)

#     matrices.append(A)
#     left_sides.append(b)
    
#     A = torch.vstack(matrices)
#     b = torch.hstack(left_sides)


#     polytop = pc.Polytope(A.numpy(), b.numpy())


#     return polytop.volume
    
