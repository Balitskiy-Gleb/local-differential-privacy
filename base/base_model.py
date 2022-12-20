
from typing import TYPE_CHECKING, List
import numpy as np
if TYPE_CHECKING:
    from configs.config import FedConfig


class BaseFedModel:
    def __init__(self, model_config, fed_config):
        self.model_config = model_config
        self.fed_config = fed_config
        self.random_ = np.random.RandomState(self.model_config.seed)
        self.model = self._init_from_config()
        self.final_update = None

    def _init_from_config(self):
        raise NotImplementedError("Please implement init method")

    def update_model_locally(self, **kwargs):
        raise NotImplementedError("Please implement update_model_locally method")

    def compute_update_one_user(self, user_id, **kwargs):
        raise NotImplementedError("Please implement compute_update method")

    def update_with_perturbed(self, update):
        raise NotImplementedError("Please implement update_with_perturbed method")

    def aggregate(self, perturbed_updates):
        raise NotImplementedError("Please implement aggregate method")

    def predict(self, user_id, **kwargs):
        raise NotImplementedError("Please predict")

    def save_model(self, path):
        raise NotImplementedError("Please implement save_model method")

    def load_model(self, path):
        raise NotImplementedError("Please implement save_model method")


