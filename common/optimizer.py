from common.model import Optimizer
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)
    
    def update(self, model):
        params = model.get_params()
        grads = model.get_grads()
        for i in range(len(params)):
            grad = grads[i]
            params[i] -= self.lr * grads[i]
            model.subtr_params(i, self.lr * grad)