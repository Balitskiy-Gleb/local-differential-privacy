import sys
import os
sys.path.insert(0,  r'../')

from base.ldp_base import BaseLDP
import numpy as np


class HarmonyLDP(BaseLDP):
    def __init__(self, eps: float = 0.5,
                 k: int = 1,
                 modified: bool = True,
                 norm_input: bool = True,
                 seed: int = 42,
                 **kwargs):
        super().__init__(eps, seed, **kwargs)
        self.k = k
        self.modified = modified
        self.norm_input = norm_input
        self.max_grads = []
        self.n_users = 0
        self.name = "e=" + str(eps) + "_k=" + str(k)

    def encode(self, data_r, eps=None):
        div_add = 10 ** (-9)
        if self.modified:
            self.max_grads.append(data_r.max())
        self.n_users += 1

        if eps is None:
            eps = self.eps

        if self.norm_input:
            low_bound = np.quantile(data_r, 0.0001) - div_add
            up_bound = np.quantile(data_r, 0.9999) + div_add
            data = data_r * 2 / (up_bound - low_bound)
        else:
            data = data_r
        flatten_data = np.clip(data.reshape(1, -1), -1, 1)
        dimension = flatten_data.shape[1]
        num_users = flatten_data.shape[0]

        flatten_perturbed_data = np.zeros(flatten_data.shape)
        # rand_indexes = self.random_.randint(0, dimension, size=(k, num_users))
        rand_indexes = self.random_.choice(np.arange(dimension), size=(1, self.k), replace=False).T
        prob_pos = (flatten_data[np.arange(num_users), rand_indexes] * \
                    (np.exp(eps) - 1) + np.exp(eps) + 1) / 2 / (np.exp(eps) + 1)
        sampled_bernoulli = self.random_.binomial(n=1, p=np.clip(prob_pos, 0, 1), size=(self.k, 1))
        sampled_sign = np.power(-1, sampled_bernoulli + 1)

        if self.modified:
            flatten_perturbed_data[np.arange(num_users), rand_indexes] = sampled_sign
        else:
            flatten_perturbed_data[np.arange(num_users), rand_indexes] = sampled_sign * dimension * \
                                                                         (np.exp(eps) + 1) / (np.exp(eps) - 1)
        return flatten_perturbed_data.reshape(data.shape)

    def normalize(self, perturbed_data):
        if self.modified:
            scale_factor = np.mean(self.max_grads) / perturbed_data.max()
            self.n_users = 0
            self.max_grads = []
            return scale_factor * perturbed_data
        else:
            return perturbed_data