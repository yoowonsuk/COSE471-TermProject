import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.model import Net
from common.layer import ParellelAffine, Affine, Sigmoid, SoftmaxWithLoss
class CustomCBOW(Net):
    def __init__(self, input_size, hidden_size, output_size, num):
        super().__init__()
        I, H, O = input_size, hidden_size, output_size
        # 계층 생성
        self.add_layers([
            ParellelAffine(I, H, num),
            Affine(H, O)
        ])
        self.loss_layer = SoftmaxWithLoss()
        self.word_vecs = self.get_params()[0]