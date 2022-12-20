import sys
sys.path.insert(0, r'.../')
from configs.grid_config import GridConfig

config = GridConfig(
    root_dir='./',
    fed_name="MovieLens_Kronecker",
    model_config="./model/configs/exps/mf_movielens_kron.py",
    model_name='MF',
    ldp_mechanisms=["Harmony", "Harmony"],
    ldp_params=[[{'eps': 1.5, 'k': 1}, {'eps': 1.5, 'k': 1}],
                  [{'eps': 1.5, 'k': 1}, {'eps': 1.5, 'k': 5}],
                  [{'eps': 1.5, 'k': 1}, {'eps': 1.5, 'k': 10}],
                  [{'eps': 1.5, 'k': 2}, {'eps': 1.5, 'k': 20}],
                  [{'eps': 1.5, 'k': 2}, {'eps': 1.5, 'k': 30}],
                  [{'eps': 1.5, 'k': 2}, {'eps': 1.5, 'k': 60}],
                  [{'eps': 1.5, 'k': 3}, {'eps': 1.5, 'k': 90}],
                  [{'eps': 1.5, 'k': 3}, {'eps': 1.5, 'k': 120}],
                  [{'eps': 1.5, 'k': 4}, {'eps': 1.5, 'k': 150}]],
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
