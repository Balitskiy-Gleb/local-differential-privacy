
import numpy as np
from typing import TYPE_CHECKING
import sys
sys.path.insert(0,  r'../')
if TYPE_CHECKING:
    from configs.config import FedConfig
from base.metrics_base import BaseScoreMeter


def init_score_meter(config: "FedConfig"):
    if config.metric == 'HR':
        return ScoreMeterRec(10)


class ScoreMeterRec(BaseScoreMeter):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        for i in range(1, topk + 1):
            self.scores["HR@" + str(i)] = 0


    def update_score(self, prediction, gt):
        args = self.topsort(np.append(gt, prediction), self.topk)
        position = np.where(args == gt.size)[0]
        if position.shape[0]:
            for t_pos in range(-self.topk, -position[0]):
                self.scores["HR@" + str(-t_pos)] += 1
        self.n_users += 1

    def compute_score(self):
        for key in self.scores.keys():
            self.scores[key] = self.scores[key] / self.n_users
        scores_ep = self.scores.copy()
        self.clear_score()
        return scores_ep

    def clear_score(self):
        for key in self.scores.keys():
            self.scores[key] = 0
        self.n_users = 0

    @staticmethod
    def topsort(a, topk):
        '''
        Compute indexes of top K buggest values


        Parameters:
        ____________

        a: numpy.ndarray
            target array
        topk: int
            number of chosen elements

        Returns:
        ________
        numpy.ndarray
            indexes of topK biggest values
        '''

        parted = np.argpartition(a, -topk)[-topk:]
        return parted[np.argsort(-a[parted])]



