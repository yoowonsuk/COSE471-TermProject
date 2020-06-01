import sys
sys.path.append('..')
from common.model import Net
from common.layer import ParellelAffine, Affine, SoftmaxWithLoss, Mean, HSLogLoss
class CustomCBOW(Net):
    def __init__(self, input_size, hidden_size, output_size, num, hs=False):
        super().__init__()
        self.hs = hs
        self.tree = None
        I, H, O = input_size, hidden_size, output_size
        if hs is False:
            self.add_layers([
                ParellelAffine(I, H, num),
                Mean(num),
                Affine(H, O)
            ])
            self.add_lossLayer([SoftmaxWithLoss()])
        else:
            self.add_layers([
                ParellelAffine(I, H, num),
                Mean(num),
                Affine(H, O-1)
            ])
            self.add_lossLayer([HSLogLoss(O-1)])
        self.word_vecs = self.get_params()[0]
    
    def get_inputw(self):
        params = self.get_params()
        return params[0], params[1]

    def get_outputw(self):
        params = self.get_params()
        return params[2], params[3]
    
    def set_HsSetting(self, id2node):
        for lossLayer in self.loss_layers:
            lossLayer.id2node = id2node