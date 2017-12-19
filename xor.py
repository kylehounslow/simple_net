import numpy as np
from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh, Relu
from joelnet.optim import SGD


inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    # Relu(),
    Linear(input_size=2, output_size=2),

])

train(net, inputs, targets, num_epochs=5000, optimizer=SGD(lr=0.05))

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
