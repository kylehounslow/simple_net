"""
Tensor
"""
import numpy as np


class Tensor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        if 'shape' not in kwargs:
            kwargs['shape'] = None
        return super(Tensor, cls).__new__(cls, *args, **kwargs)
