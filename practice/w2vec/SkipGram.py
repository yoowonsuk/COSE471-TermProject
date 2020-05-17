import sys
sys.path.append('..')
from common.model import Net
from common.layer import ParellelOutputAffine, Affine, Sigmoid, SoftmaxWithLoss