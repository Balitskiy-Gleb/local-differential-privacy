
from base.ldp_base import BaseLDP
import numpy as np

class LaplaceLDP(BaseLDP):
    def __init__(self,
                 eps: float = 0.5,
                 seed: int = 42,
                 **kwargs):
        super().__init__(eps, seed, **kwargs)
        self.MAX_VALUE = 10000
        self.name = "e=" + str(eps)

    def encode(self, data, eps=None):
        if eps is None:
            eps = self.eps

        up, low = np.quantile(data, 0.995), np.quantile(data, 0.005)
        sensitivity = (up - low)
        data = np.clip(data, low, up)
        sensitivity = np.clip(sensitivity, -self.MAX_VALUE, self.MAX_VALUE)
        if np.isnan(sensitivity):
            sensitivity = self.MAX_VALUE

        dimension = data.size
        ldp_noise = self.random_.laplace(0, dimension * sensitivity / eps,
                                         size=np.shape(data))
        return data + ldp_noise

