from typing import TYPE_CHECKING, List
import sys

sys.path.insert(0, r'.../')
from pydantic import BaseModel
import shutil
from configs.config import FedConfig, dump_config
from datetime import datetime
import  os
if TYPE_CHECKING:
    from base.base_model import BaseFedModel
    from base.ldp_base import BaseLDP
    from base.metrics_base import BaseScoreMeter

class GridConfig(BaseModel):
    root_dir = './'
    fed_name: str = "GridSearchLaplace"
    model_config: str = "./model/configs/exps/mf_movielens_small.py"
    model_name: str = 'MF'
    ldp_mechanisms = ["Laplace"]
    ldp_params = [[{'eps': 1}],
                  [{'eps': 10}],
                  [{'eps': 50}],
                  [{'eps': 100}],
                  [{'eps': 500}],
                  [{'eps': 1000}],
                  [{'eps': 5000}],
                  [{'eps': 10000}],
                  [{'eps': 15000}]]
    output_path: str = "./logs/"
    seed: int = 42

    metric: str = "HR"
    # criterion: "BaseCriterion"
    data_path: str = "./data/MovieLens/data/MovieLens_ready.csv"
    train_server_iter = 2
    lr_init = 0.01 * 5
    n_epochs = 10
    regularization = 0.001
    batch_size = 10
    optimizer = 'sgd'
    scheduler = 'stepLR'
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    # def __init__(self):
    #     super(GridConfig, self).__init__()
    #     self.generate_logs()
    #
    def generate_logs(self):
        log_path = os.path.join(self.root_dir, self.output_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_path = os.path.join(log_path, self.fed_name)

        log_path = log_path + "_" + self.time
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        return log_path
        # dump_config( ,os.path.join(log_path, "config.py"))

    def generate_item(self):
        log_path = self.generate_logs()
        for ldp_par_set in self.ldp_params:
            config = FedConfig(
                root_dir=self.root_dir,
                fed_name=self.fed_name,
                model_config=self.model_config,
                model_name=self.model_name,
                ldp_mechanisms=self.ldp_mechanisms,
                ldp_params=ldp_par_set,
                output_path=log_path,
                seed=self.seed,
                metric=self.metric,
                # criterion: "BaseCriterion"
                data_path=self.data_path,
                train_server_iter=self.train_server_iter,
                lr_init=self.lr_init,
                n_epochs=self.n_epochs,
                regularization=self.regularization,
                batch_size=self.batch_size,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                time=self.time
            )
            yield config



