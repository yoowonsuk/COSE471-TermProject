from model import Layer, LossLayer
from function import softmax, cross_entropy_error
import torch

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = 1 / (1 + torch.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        W = 0.01 * torch.randn(input_size, output_size)
        b = torch.zeros(output_size)
        self.add_params([W, b])
        self.add_grads([torch.zeros_like(W), torch.zeros_like(b)])

    def forward(self, x):
        W, b = self.get_params()
        out = torch.mm(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.get_params()
        dx = torch.mm(dout, W.T)
        dW = torch.mm(self.x.T, dout)
        db = torch.sum(dout, axis=0)

        self.set_grads(0, dW)
        self.set_grads(1, db)
        return dx

class SoftmaxWithLoss(LossLayer):
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  
        self.t = None  

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size() == self.y.size():
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y
        dx[torch.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx