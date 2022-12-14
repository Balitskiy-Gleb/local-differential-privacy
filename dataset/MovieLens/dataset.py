import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix, issparse
import pandas as pd
import numpy as np


class MovieLensDatasetLOO():
    def __init__(self, data_path, **kwargs):
        self.sparse = kwargs["sparse"]
        self.rating = kwargs["rating"]
        self.alpha = kwargs["alpha"]
        self.data_path = data_path
        self.interaction_matrix = self.get_interaction_matrix()
        self.cui = None
        self.pu = None
        self.test_indexes = None

    def update_cui_pu(self):
        if issparse(self.interaction_matrix):
            self.cui = self.interaction_matrix.copy()
            self.pu = self.interaction_matrix.copy()
            self.cui.data = self.alpha * self.cui.data + 1
            self.cui.data[self.test_indexes] = 0
            self.pu.data = (self.cui.data > 0).astype(int)
        else:
            self.cui = 1 + self.alpha * self.interaction_matrix
            self.cui[np.arange(self.cui.shape[0]), self.test_indexes] = 0
            self.pu = (self.cui.data > 0).astype(int)

    def get_interaction_matrix(self):
        data = pd.read_csv(self.data_path)
        num_interactions = data.shape[0]
        size = data.userId.unique().size, data.movieId.unique().size,
        if self.sparse:
            if self.rating:
                return csr_matrix((data.rating.values,
                                   (data.userId.values, data.movieId.values)),
                                  shape=size)
            else:
                return csr_matrix((np.ones(num_interactions),
                                   (data.userId.values, data.movieId.values)),
                                  shape=size)
        else:
            if self.rating:
                int_matrix = np.zeros(size)
                int_matrix[data.userId.values, data.movieId.values] = data.rating.values
                return int_matrix
            else:
                int_matrix = np.zeros(size, dtype=int)
                int_matrix[data.userId.values, data.movieId.values] = 1
                return int_matrix

    def generate_test(self, loo=True):
        if loo:
            if issparse(self.interaction_matrix):
                ind_ptr = self.interaction_matrix.indptr
                test_indexes = np.random.randint(ind_ptr[:-1],
                                                 ind_ptr[1:],
                                                 size=(ind_ptr.shape[0] - 1))
                return test_indexes
            else:
                user_id, movie_id = self.interaction_matrix.nonzero()
                ind_ptr = np.zeros(self.interaction_matrix.shape[0] + 1, dtype=int)
                ind_ptr[1:] = np.unique(user_id, return_counts=True)[1].cumsum()
                test_indexes = np.random.randint(ind_ptr[:-1],
                                                 ind_ptr[1:],
                                                 size=(ind_ptr.shape[0] - 1))
                test_movie_indexes = movie_id[test_indexes]
                return test_movie_indexes

    def get_item(self, **kwargs):
        self.test_indexes = self.generate_test()
        self.update_cui_pu()
        return self.cui, self.pu, self.test_indexes