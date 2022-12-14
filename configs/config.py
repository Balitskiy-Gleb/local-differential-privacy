from typing import TYPE_CHECKING, List
import sys
sys.path.insert(0, r'.../')
from pydantic import BaseModel
import shutil


if TYPE_CHECKING:
    from base.base_model import BaseFedModel
    from base.ldp_base import BaseLDP
    from base.metrics_base import BaseScoreMeter


class FedConfig(BaseModel):
    root_dir = './'
    fed_name: str = "Test"
    model_config: str = "./model/configs/exps/mf_movielens_small.py"
    model_name: str = 'MF'
    ldp_mechanisms = ["Laplace"],
    ldp_params = [{'eps': 1000000}],
    output_path: str = "./logs/"
    seed: int = 42
    metric: str = "HR"
    # criterion: "BaseCriterion"
    data_path: str = "./data/MovieLens/data/MovieLens_ready.csv"
    train_server_iter = 2
    lr_init = 0.01 * 5
    n_epochs = 2
    regularization = 0.001
    batch_size = 10
    optimizer = 'sgd'
    scheduler = 'stepLR'
    checkpoint = './exps/'
    time = ''


def dump_config(path_config, path_to_save):
    shutil.copyfile(path_config, path_to_save)


def load_config(path):
    loc = {}
    exec(open(path).read(), globals(), loc)
    config = loc["config"]
    return config





