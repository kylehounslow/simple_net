"""
Optimizers!
"""
from joelnet.nn import NeuralNet
class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
