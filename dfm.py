import numpy as np
import math
from scipy.optimize import minimize
from prediction_model import dfm

class DelayedFeedback:
    def __init__(self, use_score_cache=False):
        self.alpha = None
        self.timestamps = None
        self.feature_data = None
        self.feature_positions = None
        self.labels = None
        self.eps = 1e-9

        self.use_score_cache = use_score_cache
        self.cached_scores = None

    def fit(self, feature_data, feature_positions, labels, timestamps, maxiter=100, alpha=100):
        self.feature_data = feature_data
        self.feature_positions = feature_positions
        self.labels = labels
        self.alpha = alpha
        self.timestamps = timestamps

        # weights
        #
        # <index>          |<feature>              |
        # -----------------|-----------------------|
        # 0                | w_conversation_'bias' | <- IDX_CONVERSION_BIAS
        # 1                | w_conversation_0      | <- IDX_CONVERSION_OFFSET
        # :                | :                     |
        # 2^24             | w_conversation_2^24-1 |
        # 2^24 + 1         | w_delay_'bias'        | <- IDX_CONVERSION_BIAS
        # 2^24 + 2         | w_delay_0             | <- IDX_CONVERSION_OFFSET
        # :                | :                     |
        # (2^24 + 1) * 2   | w_delay_2^24-1        |
        len_weights = int((math.pow(2, 24) + 1) * 2)
        initial_weights = np.zeros(len_weights)

        results = minimize(self.f, initial_weights,
                method='L-BFGS-B', jac=self.g, options={"maxiter": maxiter})
        return results

    def predict(self, feature_data, feature_positions,weights):
        # TODO: there is a more efficient way cuz calc_scores calculates lambda which is not needed.
        scores = dfm.calc_scores(feature_data, feature_positions, weights) 
        return scores[:, 0]

    def g(self, weights):
        if self.use_score_cache:
            # this cache mechanism depends the internal implementation detail of scipy's l_bfgs_b.
            # g is always called after f.
            # https://github.com/scipy/scipy/blob/e5b57ce42eed88f23394c296a5283be27f6de034/scipy/optimize/lbfgsb.py#L284-L286
            scores = self.cached_scores
        else:
            scores = dfm.calc_scores(self.feature_data, self.feature_positions, weights)
        gradient = dfm.calc_gradient(self.feature_data, 
                self.feature_positions, self.labels, np.clip(scores, self.eps, 1 - self.eps), self.timestamps)
        gradient += self.alpha * weights 
        return gradient

    def f(self, weights):
        scores = dfm.calc_scores(self.feature_data, self.feature_positions, weights)
        if self.use_score_cache:
            self.cached_scores = scores

        loss = self._empirical_loss(scores=scores)

        r = 0.5 * self.alpha * np.power(np.linalg.norm(weights, 2), 2)

        return loss + r

    def _empirical_loss(self, scores):
        loss = 0

        for (label, (p, l), t) in zip(self.labels, np.clip(scores, self.eps, 1-self.eps), self.timestamps):
            loss -= int(label) * (math.log(p) + math.log(l) - l * t) + (1 - label) * math.log(1 - p + p * math.exp(-l * t))
        return loss
