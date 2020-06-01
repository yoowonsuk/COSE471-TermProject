import sys
sys.path.append('..')
from common.model import Net
from common.layer import Affine, SoftmaxWithLoss, HSLogLoss
class CustomSkipGram(Net):
    def __init__(self, input_size, hidden_size, output_size, num, hs=False):
        super().__init__()
        I, H, O = input_size, hidden_size, output_size
        if hs is False:
            self.add_layers([
                Affine(I, H),
                Affine(H, O)
            ])
            for _ in range(num):
                self.add_lossLayer([SoftmaxWithLoss()])
        else:
            self.add_layers([
                Affine(I, H),
                Affine(H, O-1)
            ])
            for _ in range(num):
                self.add_lossLayer([HSLogLoss(O-1)])
        self.word_vecs = self.get_params()[0]
        self.num = num
    
    def get_inputw(self):
        params = self.get_params()
        return params[0], params[1]

    def get_outputw(self):
        params = self.get_params()
        return params[2], params[3]

    def forward(self, x, t): # overloading
        score = self.predict(t)
        loss = 0
        for i, loss_layer in enumerate(self.loss_layers):
            loss += loss_layer.forward(score, x[:, i])
        return loss

    def set_HsSetting(self, id2node):
        for lossLayer in self.loss_layers:
            lossLayer.id2node = id2node