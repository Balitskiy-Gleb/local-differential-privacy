import sys
import os

sys.path.insert(0, r'..../')
from model.configs.mf_recommender_config import MFRecommenderConfig

config = MFRecommenderConfig(
    global_sigma=0.01,
    local_sigma=0.01,
    n_factors=5,
    n_items=9953,
    n_users=74657,
    seed=42,
    do_kron=True
)