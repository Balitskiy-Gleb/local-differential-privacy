import sys
import os

sys.path.insert(0, r'..../')
from model.configs.mf_recommender_config import MFRecommenderConfig

config = MFRecommenderConfig(
    global_sigma=0.01,
    local_sigma=0.01,
    n_factors=20,
    n_items=1241,
    n_users=1128,
    seed=42,
    do_kron=True
)


