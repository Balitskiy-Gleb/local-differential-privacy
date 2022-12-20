from typing import TYPE_CHECKING, List
import sys
import os
sys.path.insert(0,  r'../')
if TYPE_CHECKING:
    from configs.config import FedConfig
    from base.base_model import BaseFedModel
    from base.ldp_base import BaseLDP
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from configs.config import dump_config, load_config
from utils.metrics import init_score_meter
from ldp.laplace import LaplaceLDP
from ldp.harmony import HarmonyLDP
from ldp.non_private import NonPrivateLDP
from model.mf_recommender import MFRecommender


def init_ldp(config: "FedConfig"):
    ldp_mechs = []
    for ldp_i, ldp_name in enumerate(config.ldp_mechanisms):
        print(config.ldp_params[ldp_i])
        if ldp_name == 'Laplace':
            ldp_mechs.append(LaplaceLDP(**config.ldp_params[ldp_i]))
        if ldp_name == 'Harmony':
            ldp_mechs.append(HarmonyLDP(**config.ldp_params[ldp_i]))
        if ldp_name == 'NP':
            ldp_mechs.append(NonPrivateLDP(**config.ldp_params[ldp_i]))
    return ldp_mechs


class Federation:
    def __init__(self,
                 config: "FedConfig"):
        self.config = config
        print(config)
        self.model_config = load_config(self.config.model_config)
        self.model = MFRecommender(self.model_config, self.config)
        self.metrics = init_score_meter(self.config)
        self.ldp_mechanisms = init_ldp(self.config)
        # self.model = model
        # self.criterion = criterion
        # self.metrics = metrics
        # self.ldp_mechanisms = ldp_mechanisms
        self.perturbed_updates = [None] * len(self.ldp_mechanisms)
        self.temp_updates = [None] * len(self.ldp_mechanisms)
        self.scores = []
        self.time = datetime.now()

    def load_model(self, exp_dir: str):
        self.model.load_model(os.path.join(exp_dir, "checkpoint"))

    def train_one_epoch(self, **kwargs):
        for user_id in tqdm(range(self.model.model_config.n_users), desc="Train User Id: "):
            user_updates = self.model.compute_update_one_user(user_id, **kwargs)
            for up_i, user_up in enumerate(user_updates):
                self.temp_updates[up_i] = self.ldp_mechanisms[up_i].encode(user_up)
                self.perturbed_updates[up_i] = self.ldp_mechanisms[up_i].decode(self.temp_updates[up_i])
            self.model.aggregate(self.perturbed_updates)
        for up_i, per_up in enumerate(self.perturbed_updates):
            self.perturbed_updates[up_i] = self.ldp_mechanisms[up_i].normalize(per_up)
        self.model.update_with_perturbed(self.perturbed_updates)

    def train(self, **kwargs):
        for epoch in tqdm(range(self.config.n_epochs), desc="Epoch"):
            self.model.update_model_locally(**kwargs)
            for i in range(self.config.train_server_iter):
                self.train_one_epoch(**kwargs)
            self.validate(**kwargs)
        self.end_train()

    def validate(self, **kwargs):
        for user_id in tqdm(range(self.model.model_config.n_users), desc="Val User Id: "):
            prediction, gt = self.model.predict(user_id, **kwargs)
            self.metrics.update_score(prediction, gt)
        self.scores.append(self.metrics.compute_score())
        print("Scores:", self.scores[-1])

    def end_train(self):
        log_path = self.config.output_path
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        final_scores_path = os.path.join(log_path, "scores.csv")

        if os.path.exists(final_scores_path):
            final_scores_df = pd.read_csv(final_scores_path)
            final_scores_df = final_scores_df.append(self.scores[-1], ignore_index=True)
            final_scores_df.to_csv(final_scores_path, index=False)
        else:
            final_scores_df = pd.DataFrame()
            final_scores_df = final_scores_df.append(self.scores[-1], ignore_index=True)
            final_scores_df.to_csv(final_scores_path, index=False)

        score_df = pd.DataFrame()
        name = ''
        for ldp in self.ldp_mechanisms:
            name += ldp.name + "_"
        name = name[:-1]
        log_path = os.path.join(log_path,name )
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        print(os.path.abspath(log_path))
        for epoch, score in enumerate(self.scores):
            score_df = score_df.append(score, ignore_index=True)
        score_df.to_csv(os.path.join(log_path, 'scores.csv'), index=False)
        dump_config(self.config.model_config, os.path.join(log_path, "config.py"))
        self.model.save_model(os.path.join(log_path, "checkpoint"))


























