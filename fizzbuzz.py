from typing import List
import numpy as np
import pandas as pd
from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh
from joelnet.optim import SGD
from joelnet.data import BatchIterator


def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    if x % 5 == 0:
        return [0, 0, 1, 0]
    if x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

def binary_encode(x: int, num_bits: int = 10) -> List[int]:
    """
    10 digit binary encoding of x
    :param x:
    :return:
    """
    return [x >> i & 1 for i in range(num_bits)]
NUM_ENCODE_BITS = 10
NUM_EPOCHS = 10000
inputs = np.array([
    binary_encode(x, NUM_ENCODE_BITS)
    for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=NUM_ENCODE_BITS, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net=net,
      inputs=inputs,
      targets=targets,
      num_epochs=NUM_EPOCHS,
      optimizer=SGD(lr=0.001))

for x in range(1, 101):
    predicted = net.forward(inputs=binary_encode(x))
    predicted_idx = np.argmax(predicted) # largest value is predicted class
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), 'fizz', 'buzz', 'fizzbuzz']
    print(x, labels[predicted_idx], labels[actual_idx])

