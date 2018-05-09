"""
CRF(Linear-Chain CRF）

Reference: 高村, 自然言語処理シリーズ1 言語処理のための機械学習入門 コロナ社 (2010)

"""

import numpy as np
from scipy.special import logsumexp
from numpy.linalg import norm

class Crf(object):
    """ CRF sample implematation.

    This class uses L2 regularization for fit.

    Parameters:
        num_tokens: int
            The number of token type.
        num_labels: int
            The number of label type.
        feature_func: callable
            The feature function. It has a signature below.
            Args:
               x: encoded token sequence
               y: encoded label
               yprev: previous encoded label
            Return:
               ndarray
        num_features: int
            the size of feature_func's return.
        lr: float
            learming rate.
        reg: float
            regularization strength.
        random_state: int
            a seed for numpy.random.
    
    """

    def __init__(self, num_tokens, num_labels, feature_func, num_features, lr,
                 reg, random_state=1):
        np.random.seed(seed=random_state)
        self.num_tokens = num_tokens
        self.num_labels = num_labels
        self.feature_func = feature_func
        self.w = 0.01 * np.random.randn(num_features)
        self.lr = lr
        self.reg = reg

    def fit(self, x, y):
        """Learn the CRF model according to the given sequence data.

        Args:
           x: ndarray, shape(n_steps)
              Sequential encoded data.
           y ndarray shape(n_steps)
              A Target encoded label sequence.
        """
        seq_len = len(x)

        log_alpha = self.calc_log_alpha(seq_len, x)
        log_beta = self.calc_log_beta(seq_len, x)
        log_distribution = logsumexp(np.asarray(
            [self.log_energy(x, '_END_', yi) + log_alpha[-1, yi]
             for yi in range(self.num_labels)]))

        expectation = np.zeros(self.w.shape)
        for yi in range(self.num_labels):
            expectation += np.exp(self.log_energy(x, yi, '_BEGIN_')
                                  + log_beta[0, yi]
                                  - log_distribution) * self.feature_func(x, yi, '_BEGIN_')
        for t in range(1, seq_len):
            for yi in range(self.num_labels):
                for yj in range(self.num_labels):
                    expectation += np.exp(self.log_energy(x, yi, yj, t)
                                          + log_alpha[t-1, yj]
                                          + log_beta[t, yi]
                                          - log_distribution) * self.feature_func(x, yi, yj, t)
        for yi in range(self.num_labels):
            expectation += np.exp(self.log_energy(x, '_END_', yi)
                                  + log_alpha[-1, yi]
                                  - log_distribution) * self.feature_func(x, '_END_', yi)

        data_feature = self.feature_func(x, y[0], '_BEGIN_')
        data_feature += np.sum(np.asarray(
            [self.feature_func(x, y[t], y[t-1], t)
             for t in range(1, seq_len)]), axis=0)
        data_feature += self.feature_func(x, '_END_', y[-1])
        self.w -= self.lr * (expectation - data_feature
                             + self.reg * norm(self.w, 2))

    def log_energy(self, x, y, yprev, idx=None):
        return self.w.dot(self.feature_func(x, y, yprev, idx))

    def calc_log_alpha(self, seq_len, x):
        log_alpha = np.zeros((seq_len, self.num_labels))
        log_alpha[0] = np.asarray([self.log_energy(x, yi, '_BEGIN_')
                                   for yi in range(self.num_labels)])
        for t in range(1, seq_len):
            for yi in range(self.num_labels):
                log_alpha[t, yi] = logsumexp(np.asarray(
                    [self.log_energy(x, yi, yj, t) + log_alpha[t-1, yj]
                     for yj in range(self.num_labels)]))
        return log_alpha

    def calc_log_beta(self, seq_len, x):
        log_beta = np.zeros((seq_len, self.num_labels))
        log_beta[-1] = np.asarray([self.log_energy(x, '_END_', yi)
                                   for yi in range(self.num_labels)])
        for t in range(seq_len-2, -1, -1):
            for yi in range(self.num_labels):
                log_beta[t, yi] = logsumexp(np.asarray(
                    [self.log_energy(x, yj, yi, t) + log_beta[t+1, yj]
                     for yj in range(self.num_labels)]))
        return log_beta

    def predict(self, x):
        """predict a label sequence for the given token sequence.

        Args:
            x: ndarray, shape (n_steps)
               Target encoded token sequence.

        Return:
           ndarray, shape(num_tokens)
           An encoded label sequence.
        """

        seq_len = len(x)
        max_val = np.zeros((seq_len + 2, self.num_labels))
        max_prev_idx = np.zeros((seq_len + 2, self.num_labels),
                                dtype=np.uint64)

        for yi in range(self.num_labels):
            vals = np.asarray(
                [self.w.dot(self.feature_func(x, yi, '_BEGIN_'))])
            max_prev_idx[1, yi] = np.argmax(vals)
            max_val[1, yi] = vals[max_prev_idx[1, yi]]
        for t in range(2, seq_len+1):
            for yi in range(self.num_labels):
                vals = np.asarray([self.w.dot(
                    self.feature_func(x, yi, yj, t-1))
                                   + max_val[t-1, yj]
                                   for yj in range(self.num_labels)])
                max_prev_idx[t, yi] = np.argmax(vals)
                max_val[t, yi] = vals[max_prev_idx[t, yi]]
        for yi in range(self.num_labels):
            vals = np.asarray([self.w.dot(
                self.feature_func(x, '_END_', yj))
                               + max_val[-2, yj]
                               for yj in range(self.num_labels)])
            max_prev_idx[-1, yi] = np.argmax(vals)
            max_val[-1, yi] = vals[max_prev_idx[-1, yi]]

        result = np.zeros(seq_len+2, dtype=np.uint64)
        result[-1] = np.argmax(max_val[-1])
        for t in range(seq_len, -1, -1):
            result[t] = max_prev_idx[t+1, result[t+1]]
        return result[1:-1]
