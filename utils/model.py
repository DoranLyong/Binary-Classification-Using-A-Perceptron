import torch 

from .functions import sigmoid, grad_sigmoid, bce_loss, grad_bce_loss


def linear_model(W, x, b): 
    """
    Neuron: Wx + b
    """
    return torch.matmul(x, W) + b


def grad_linear_model(x):
    """
    Derivative of y = WX + B with regard to both parameters;  W and b
    """
    batch_size = x.shape[0]
    g_w = x     # dy/dW = x
    g_b = torch.ones(batch_size)  # dy/db = 1

    return g_w, g_b





class BinaryClassifierGraph:
    def __init__(self): 
        """
        It is initializing the variable that will be updated in `forward` and `loss` function. 
        Storing these values will be used in `backward` function to get the gradient.

        Model: y= w0*x0 + w1*x1 + b 
        """

        # default gardient is zero
        self.w0_grad = 0     
        self.w1_grad = 0
        self.b_grad = 0
        
        self.x_in = None
        self.wx_plus_b_out = None  
        self.sigmoid_out = None
        
        self.bce_loss = None
        
        self.grad_bce_loss = None
        
        self.g_truth = None        

    
    def forward(self, w, x, b):
    
        # updated input value, it will be used in backward pass
        self.x_in = x
        self.b_in = b
        self.w_in = w
        
        # Intermediate node with the weighted sum
        self.wx_plus_b_out = linear_model(w, x, b)
        
        # Output node after applying activation function
        self.sigmoid_out = sigmoid(self.wx_plus_b_out)
            
        return self.sigmoid_out


    def loss(self, g_truth): 

        self.g_truth = g_truth
        
        # Compute the binary cross entropy loss
        self.bce_loss = bce_loss(self.sigmoid_out, g_truth)
        return self.bce_loss.mean()    



    def backward(self): 
        # Compute the gradients of Loss w.r.t neuron output (y')
        d_bce_loss = grad_bce_loss(self.sigmoid_out, self.g_truth)
        
        # Compute the gradients of neuron output(y') w.r.t weighted sum(z)
        d_sigmoid = grad_sigmoid(self.wx_plus_b_out)
        
        # Compute the gradients of weighted sum(z) w.r.t weights and bias
        d_w, d_b = grad_linear_model(self.x_in)
        
        # Using chain rule to find overall gradient of Loss w.r.t weights and bias
        self.w0_grad = d_bce_loss * d_sigmoid * d_w[:,0]
        self.w1_grad = d_bce_loss * d_sigmoid * d_w[:,1]
        self.b_grad = d_bce_loss * d_sigmoid * d_b
        
        return        

    
    def gradients(self):
        
        w_grad = torch.tensor([[self.w0_grad.mean()], [self.w1_grad.mean()]])
        b_grad = torch.tensor([self.b_grad.mean()])
        
        return w_grad, b_grad