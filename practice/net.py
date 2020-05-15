from model import Net
from layer import Affine, Sigmoid, SoftmaxWithLoss
class TwoLayerNet(Net):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        I, H, O = input_size, hidden_size, output_size

        # 계층 생성
        self.layers = [
            Affine(I, H),
            Sigmoid(),
            Affine(H, O)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        for layer in self.layers:
            self.add_params(layer.get_params())
            self.add_grads(layer.get_grads())