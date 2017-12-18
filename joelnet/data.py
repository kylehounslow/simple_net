"""
generate batches of data to feed through neural net
"""

from typing import Iterator, NamedTuple

import numpy as np

from joelnet.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets" Tensor)])

class DataIterator:
    pass # TODO: at 31minute mark