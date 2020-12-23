import torch 


def sigmoid(x):
    return torch.sigmoid(x).squeeze()  # (ref) https://pytorch.org/docs/stable/generated/torch.squeeze.html

def grad_sigmoid(x):
    """
    Derivative of sigmoid(x) with regrad to the parameter x
    """
    return sigmoid(x) * (1 - sigmoid(x))


def bce_loss(sigmoid_pred, g_truth):
    """
    Binary Cross-Entropy Loss (= logistic loss)
    """
    loss = - (1 - g_truth)* torch.log(1 - sigmoid_pred) - g_truth * torch.log(sigmoid_pred)
    return loss 

def grad_bce_loss(sigmoid_pred, g_truth):
    """
    Derivative of BCE with regard to its input. 
    """
    return - (g_truth * (1 / sigmoid_pred)) + ((1 - g_truth) * (1 / (1 - sigmoid_pred)))


