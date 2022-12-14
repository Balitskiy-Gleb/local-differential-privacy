from base.ldp_base import BaseLDP
import numpy as np


class NonPrivateLDP(BaseLDP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MAX_VALUE = 10000

    def encode(self, data, eps=None):
        return data