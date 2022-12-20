import argparse
from dataset.MovieLens.dataset import MovieLensDatasetLOO
from configs.config import FedConfig, load_config
from configs.grid_config import GridConfig
from model.mf_recommender import MFRecommender
from model.configs.mf_recommender_config import MFRecommenderConfig
from ldp.laplace import LaplaceLDP
from ldp.harmony import HarmonyLDP
from utils.metrics import ScoreMeterRec
from federation.federation import Federation


def init_ldp(config: FedConfig):
    ldp_mechs = []
    for ldp_i, ldp_name in enumerate(config.ldp_mechanisms):
        if ldp_name == 'Laplace':
            ldp_mechs.append(LaplaceLDP(**config.ldp_params[ldp_i]))
        if ldp_name == 'Harmony':
            ldp_mechs.append(HarmonyLDP(**config.ldp_params[ldp_i]))
    return ldp_mechs


def main():
    parser = argparse.ArgumentParser(description='Train Federation System')
    parser.add_argument("--cfg", type=str, default='./configs/exps/gs_mf_movielens_small.py')
    args = parser.parse_args()
    config = load_config(args.cfg)
    if isinstance(config, GridConfig):
        for config_exp in config.generate_item():
            federation = Federation(config_exp)

            params = {"sparse": True,
                      "rating": False,
                      "alpha": 1}
            data = MovieLensDatasetLOO(config.data_path, **params)
            cui, pu, test_ind = data.get_item()
            train_params = {"Cui": cui, "Pu": pu, "test_indexes": test_ind}
            federation.train(**train_params)
            del federation
    else:
        federation = Federation(config)

        params = {"sparse": True,
                  "rating": False,
                  "alpha": 1}
        data = MovieLensDatasetLOO(config.data_path, **params)
        cui, pu, test_ind = data.get_item()
        train_params = {"Cui": cui, "Pu": pu, "test_indexes": test_ind}
        federation.train(**train_params)

if __name__ == '__main__':
    main()
