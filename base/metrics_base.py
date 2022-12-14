
class BaseScoreMeter:
    def __init__(self, **kwargs):
        self.scores = {}
        self.n_users = 0

    def update_score(self, prediction, gt):
        raise NotImplementedError

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

