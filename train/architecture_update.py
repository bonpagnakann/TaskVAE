import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

def update_weights(net, amnt_new_classes):
    layer_key = list(net.state_dict().keys())[-2]
    weights = net.state_dict()[layer_key].cpu().detach().numpy()
    w_mean = np.mean(weights, axis=0)
    w_std = np.std(weights, axis=0)
    new_weights = np.pad(weights, ((0, amnt_new_classes), (0, 0)), mode="constant", constant_values=0)
    for i in reversed(range(amnt_new_classes)):
        for j in range(new_weights.shape[1]):
            new_weights[new_weights.shape[0] - 1 - i][j] = np.random.normal(w_mean[j], w_std[j])
    return new_weights


def update_bias(net, amnt_new_classes):
    bias_key = list(net.state_dict().keys())[-1]
    bias = net.state_dict()[bias_key].cpu().detach().numpy()
    b_mean = np.mean(bias)
    b_std = np.std(bias)
    new_bias = np.zeros(len(bias) + amnt_new_classes, dtype="f")
    new_bias[:len(bias)] = bias
    for i in range(amnt_new_classes):
        new_bias[-1 - i] = np.random.normal(b_mean, b_std) - np.log(amnt_new_classes)
    return new_bias

def custom_init(m, model):
    if isinstance(m, nn.Conv1d):
        # Kaiming Normal for Conv1d layers with ReLU
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    elif isinstance(m, nn.Linear):
        if m is not list(model.modules())[-1]:
            # Kaiming Normal for Linear layers with ReLU
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        else:
            # Xavier Normal for the final Linear layer with Softmax
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def update_model(NET_FUNCTION, net, amnt_new_classes, device):
    amnt_old_classes = list(net.children())[-1].out_features

    in_features = list(net.children())[-1].in_features
    out_features = amnt_old_classes
    weights = net.fc.weight.data
    new_out_features = out_features + amnt_new_classes
    net.fc = torch.nn.Linear(in_features,
                             new_out_features, bias=False)
    custom_init(net.fc, net)
    net.fc.weight.data[:out_features] = weights
    net = net.to(device)

    return net

def check_model_integrity(old_model, new_model, verbose=False):
    for i in old_model.state_dict().keys():
        if (np.array_equal(old_model.state_dict()[i].cpu().numpy(), new_model.state_dict()[i].cpu().numpy())):
            if verbose:
                print(f"key {i} is the same for both nets")
        else:
            if verbose:
                print("\n", i, "\n")
            for h in range(len(old_model.state_dict()[i])):
                try:
                    if np.array_equal(old_model.state_dict()[i][h].numpy(), new_model.state_dict()[i][h].numpy()):
                        if verbose:
                            print(f"key {i} weights of neuron {h} are the same for both nets\n")
                    else:

                        print(f"key {i} weights of neuron {h} are different for both nets\n Differces at:")
                        print(old_model.state_dict()[i][h].numpy() - new_model.state_dict()[i][h].numpy())
                        print("\n")
                        return False
                except:
                    pass
    return True
