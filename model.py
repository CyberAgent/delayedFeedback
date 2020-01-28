import numpy as np
import math
from scipy.optimize import minimize
from prediction_model import dfm
from sklearn.metrics import log_loss, roc_auc_score

class DFM:
    """Delayed Feedback Model introducted by Chapelle(2014)"""

    def __init__(self, alpha=10, maxiter=100, eps=1e-9, use_score_cache=False):
        self.alpha = None
        self.timestamps = None
        self.alpha = alpha
        self.maxiter = maxiter
        self.eps = eps

        # self.coef_
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
        self.coef_ = None

        self.use_score_cache = use_score_cache
        self.cached_scores = None

    def fit(self, X, X_positions, num_features, y, timestamps):
        self.X = X
        self.X_positions = X_positions
        self.num_features = num_features
        self.y = y
        self.timestamps = timestamps
        initial_coef_ = np.zeros((num_features + 1) * 2)

        result = minimize(self.loss_func, initial_coef_,
                method='L-BFGS-B', jac=self.g, options={"maxiter": self.maxiter})
        self.coef_ = result['x']
        return self

    def predict(self, X, X_positions):
        # TODO: there is a more efficient way because calc_scores calculates lambda which is not needed.
        scores = dfm.calc_scores_dfm(X, X_positions, self.num_features, self.coef_) 
        return scores[:, 0]

    def dump(self, f):
        import pickle
        with open(f, "wb") as f:
            pickle.dump(self, f)

    def score(self, X, X_positions, y):
        scores = dfm.calc_scores_dfm(X, X_positions, self.num_features, self.coef_) 
        predicted_value = scores[:, 0]
        return log_loss(y, predicted_value), roc_auc_score(y, predicted_value)

    def g(self, coef_):
        if self.use_score_cache:
            # this cache mechanism depends the internal implementation detail of scipy's l_bfgs_b.
            # g is always called after f.
            # https://github.com/scipy/scipy/blob/e5b57ce42eed88f23394c296a5283be27f6de034/scipy/optimize/lbfgsb.py#L284-L286
            scores = self.cached_scores
        else:
            scores = dfm.calc_scores_dfm(self.X, self.X_positions, self.num_features, coef_)
        gradient = dfm.calc_gradient_dfm(self.X, self.X_positions, self.num_features,  
                self.y.astype(bool), np.clip(scores, self.eps, 1 - self.eps), self.timestamps)
        gradient += self.alpha * coef_ 
        return gradient

    def loss_func(self, coef_):
        scores = dfm.calc_scores_dfm(self.X, self.X_positions, self.num_features, coef_)
        if self.use_score_cache:
            self.cached_scores = scores

        loss = self._empirical_loss(scores=scores)

        r = 0.5 * self.alpha * np.power(np.linalg.norm(coef_, 2), 2)

        return loss + r

    def _empirical_loss(self, scores):
        loss = 0

        for (label, (p, l), t) in zip(self.y, np.clip(scores, self.eps, 1-self.eps), self.timestamps):
            loss -= int(label) * (math.log(p) + math.log(l) - l * t) + (1 - label) * math.log(1 - p + p * math.exp(-l * t))
        return loss
