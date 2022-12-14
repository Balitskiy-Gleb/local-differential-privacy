
from typing import TYPE_CHECKING, List
import numpy as np

class BaseLDP:
    def __init__(self, eps, seed, **kwargs):
        self.random_ = np.random.RandomState(seed)
        self.MAX_VALUE = 10000
        self.eps = eps

    def init_LDP(self):
        raise NotImplementedError

    def encode(self, data, eps=None):
        raise NotImplementedError

    def decode(self, perturbed_data):
        return perturbed_data

    def normalize(self, perturbed_data):
        return perturbed_data