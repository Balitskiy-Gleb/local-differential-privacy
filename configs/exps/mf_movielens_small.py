import sys
import os

sys.path.insert(0, r'.../')
from configs.config import FedConfig

config = FedConfig(
    root_dir='./',
    fed_name="Movie Small",
    model_config="./model/configs/exps/mf_movielens_small.py",
    model_name='MF',
    ldp_mechanisms=["Laplace"],
    ldp_params=[{'eps': 1000000}],
    output_path="logs/",
    seed=42,
    metric="HR",
    # criterion: "BaseCriterion"
    data_path="./dataset/MovieLens/data/MovieLensSmall.csv",
    train_server_iter=2,
    lr_init=0.01 * 5,
    n_epochs=2,
    regularization=0.001,
    batch_size=10,
    optimizer='sgd',
    scheduler='stepLR'
)

