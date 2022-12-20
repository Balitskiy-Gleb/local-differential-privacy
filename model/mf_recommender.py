import numpy as np
from typing import TYPE_CHECKING
import sys
import os
sys.path.insert(0,  r'../')
if TYPE_CHECKING:
    from configs.config import FedConfig
    from configs.mf_recommender_config import MFRecommenderConfig
from base.base_model import BaseFedModel
from scipy.sparse import issparse
from tqdm import tqdm


class MFRecommender(BaseFedModel):
    def __init__(self, model_config: "MFRecommenderConfig", fed_config: "FedConfig"):
        self.global_model = None
        self.local_model = None
        super().__init__(model_config, fed_config)


    def _init_from_config(self):
        self.global_model = self.random_.normal(0, self.model_config.global_sigma,
                                                size=(self.model_config.n_factors,
                                                self.model_config.n_items))
        self.local_model = self.random_.normal(0, self.model_config.local_sigma,
                                               size=(self.model_config.n_factors,
                                                    self.model_config.n_users))
        self.model = (self.global_model, self.local_model)
        self.final_update = np.zeros((self.model_config.n_factors,
                                                self.model_config.n_items))

    def update_model_locally(self, **kwargs):
        Cui, Pu = kwargs["Cui"], kwargs["Pu"]
        if issparse(Cui):
            for user_id in tqdm(range(self.model_config.n_users), desc="Local update user id"):

                coef_inx = Cui.indices[Cui.indptr[user_id]: Cui.indptr[user_id + 1]]
                coef = Cui.data[Cui.indptr[user_id]: Cui.indptr[user_id + 1]] - 1
                global_matrix_nnz = self.global_model[:, coef_inx]

                left_hs = self.global_model @ self.global_model.T + (global_matrix_nnz * coef) @ global_matrix_nnz.T
                left_hs = left_hs + self.fed_config.regularization * np.eye(self.model_config.n_factors)

                pu_coef = Pu.data[Pu.indptr[user_id]: Pu.indptr[user_id + 1]]
                right_hs = (global_matrix_nnz * (coef + 1)) @ pu_coef
                self.local_model[:, user_id] = np.linalg.solve(left_hs, right_hs)

    def compute_update_one_user(self, user_id, **kwargs):

        Cui, Pu = kwargs["Cui"], kwargs["Pu"]
        if issparse(Cui):
            cui_ind = Cui.indices[Cui.indptr[user_id]: Cui.indptr[user_id + 1]]
            cui_coef = Cui.data[Cui.indptr[user_id]: Cui.indptr[user_id + 1]]
            update_part = - self.local_model[:, user_id] @ self.global_model
            pu_coef = Pu.data[Pu.indptr[user_id]: Pu.indptr[user_id + 1]]
            update_part[cui_ind] = cui_coef * (pu_coef + update_part[cui_ind])
            if self.model_config.do_kron:
                return [np.outer(self.local_model[:, user_id], -2 * update_part)]
            else:
                return [self.local_model[:, user_id], -2 * update_part]
        else:
                raise NotImplementedError("Not Sparse is Not Implemented")

    def update_with_perturbed(self, update):
        self.global_model = self.global_model - self.fed_config.lr_init * \
                        (self.fed_config.regularization * self.global_model + self.final_update)
        self.final_update = self.final_update * 0.0

    def aggregate(self, perturbed_updates):
        if self.model_config.do_kron:
            self.final_update = self.final_update + perturbed_updates[0]
        else:
            self.final_update = self.final_update + np.outer(perturbed_updates[0], perturbed_updates[1])

    def predict(self, user_id, **kwargs):
        Cui, Pu = kwargs["Cui"], kwargs["Pu"]
        test_indexes = kwargs["test_indexes"]
        global_matrix = self.global_model
        local_matrix = self.local_model
        all_movieIds = np.arange(global_matrix.shape[1])
        interaction_ind = Pu.indices[Pu.indptr[user_id]: Pu.indptr[user_id + 1]]
        not_interacted_ind_sample = np.random.choice(np.setdiff1d(all_movieIds, interaction_ind),
                                                     size=99,
                                                     replace=False)
        target_item_prediction = local_matrix[:, user_id] @ global_matrix[:, Pu.indices[test_indexes[user_id]]]
        test_item_prediction = local_matrix[:, user_id] @ global_matrix[:, not_interacted_ind_sample]
        return target_item_prediction, test_item_prediction

    def save_model(self, path):

        with open(path + ".npy", 'wb') as ckpt:
            np.save(ckpt, self.global_model)
            np.save(ckpt, self.local_model)

    def load_model(self, path):
        with open(path + ".npy", 'rb') as ckpt:
            self.global_model = np.load(ckpt)
            self.local_model = np.load(ckpt)


