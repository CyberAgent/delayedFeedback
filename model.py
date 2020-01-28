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

class DLADF():
    """Dual Learning Algorithm for Delayed Feedback introduced Saito(2019)"""

    def __init__(self, num_features, lam=0.02, beta=0., maxiter=1000, num_iter=100):
        self.num_features = num_features
        self.lam = lam
        self.beta = beta
        self.cvr_coef_ = np.zeros(num_features + 1) # np.random.normal(0, 0.01, size=self.num_features + 1)  
        self.ips_coef_ = np.zeros(num_features + 1) # np.random.normal(0, 0.01, size=self.num_features + 1) 
        self.maxiter = maxiter
        self.num_iter = num_iter

    def fit(self, X, X_positions, y):
        self.X = X
        self.X_positions = X_ositions
        self.y = y.astype(np.uint16)
        for _ in range(self.maxiter):
            update_cvr = minimize(self.loss_cvr, self.cvr_coef_, method="L-BFGS-B", jac=self.grad_cvr, options={"maxiter": self.num_iter})
            diff_cvr_coef_ = ((self.cvr_coef_ - update_cvr['x']) ** 2).sum()
            self.cvr_coef_ = update_cvr['x']

            update_ips = minimize(self.loss_ips, self.ips_coef_, method="L-BFGS-B", jac=self.grad_ips, options={"maxiter": self.num_iter})
            diff_ips_coef_ = ((self.ips_coef_ - update_ips['x']) ** 2).sum()
            self.ips_coef_ = update_ips['x']

            if (diff_ips_coef_ + diff_cvr_coef_ < 1e-11):
                break

        return self

    def _predict(self, X, X_positions, coef_):
        preds = dfm.predict(X, X_positions, coef_)
        preds = np.clip(preds, 1e-9, 1 - 1e-9)
        return preds 

    def predict(self, X, X_positions):
        """predict the conversion rate"""
        preds = self._predict(X, X_positions, self.cvr_coef_)
        return preds

    def score(self, X, X_positions, y):
        pred = dfm.predict(X, X_positions, self.cvr_coef_)
        nll = log_loss(y, pred)
        auc = roc_auc_score(y, pred)
        return nll, auc
    
    def dump(self, f):
        import pickle
        with open(f, "wb") as f:
            pickle.dump(self, f)


    def loss(self, x1, x2, y):
        ill = - y * ( np.log(x1) / x2 + ( 1.0 - 1.0 / x2 ) * np.log(1 - x1) ) - ( 1 - y ) * np.log(1 - x1)
        ill = np.clip(ill, -self.beta, 10e5).sum()
        ill = ill / (self.X_positions.shape[0] - 1)
        ill = ill + self.lam * ( x1 ** 2 ).sum()
        if np.isnan(ill):
            raise ("Loss is Nan!")
        return ill.sum()
        
    def loss_cvr(self, coef_):
        """log loss for cvr weighted with ips"""
        ps = self._predict(self.X, self.X_positions, self.ips_coef_)
        cvr = self._predict(self.X, self.X_positions, coef_)
        ill = self.loss(cvr, ps, self.y)
        return ill

    def loss_ips(self, coef_):
        cvr = self._predict(self.X, self.X_positions, self.cvr_coef_)
        ips = self._predict(self.X, self.X_positions, coef_)
        ill = self.loss(ips, cvr, self.y)
        return ill

    def grad_cvr(self, coef_):
        gradients_cvr = dfm.calc_gradient_dl(self.num_features, self.X, self.X_positions, self.y.astype(bool), coef_, self.ips_coef_)
        gradients_cvr = gradients_cvr / (self.X_positions.shape[0] - 1)
        gradients_cvr +=  2 * self.lam * coef_
        return gradients_cvr

    def grad_ips(self, coef_):
        gradients_ips = dfm.calc_gradient_dl(self.num_features, self.X, self.X_positions, self.y.astype(bool), coef_, self.cvr_coef_)
        gradients_ips = gradients_ips / (self.X_positions.shape[0] - 1)
        gradients_ips += 2 * self.lam * coef_
        return gradients_ips 
