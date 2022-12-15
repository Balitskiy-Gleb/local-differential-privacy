import sys
sys.path.insert(0, r'.../')
from configs.grid_config import GridConfig

config = GridConfig(
    root_dir='./',
    fed_name="Movie Full Harmony",
    model_config="./model/configs/exps/mf_movielens_full.py",
    model_name='MF',
    # ldp_mechanisms=["Laplace"],
    # ldp_params=[[{'eps': 1}],
    #               [{'eps': 10}],
    #               [{'eps': 50}],
    #               [{'eps': 100}],
    #               [{'eps': 500}],
    #               [{'eps': 1000}],
    #               [{'eps': 5000}],
    #               [{'eps': 10000}],
    #               [{'eps': 15000}]],
    ldp_mechanisms=["Harmony"],
    ldp_params=[[{'eps': 1.5, 'k': 1}],
                  [{'eps': 1.5, 'k': 2}],
                  [{'eps': 1.5, 'k': 10}],
                  [{'eps': 1.5, 'k': 50}],
                  [{'eps': 1.5, 'k': 100}],
                  [{'eps': 1.5, 'k': 200}],
                  [{'eps': 1.5, 'k': 400}],
                  [{'eps': 1.5, 'k': 600}],
                  [{'eps': 1.5, 'k': 1000}]],
    output_path="logs/",
    seed=42,
    metric="HR",
    # criterion: "BaseCriterion"
    data_path="./dataset/MovieLens/data/MovieLens_ready.csv",
    train_server_iter=2,
    lr_init=0.01 * 5,
    n_epochs=10,
    regularization=0.001,
    batch_size=10,
    optimizer='sgd',
    scheduler='stepLR'
)
