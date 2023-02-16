import numpy as np
from torch.distributions import Categorical, MultivariateNormal, Normal, \
    LowRankMultivariateNormal, kl_divergence

RESAMPLE_THRESH = 0.5
SIGMA = 20.0
P_INIT = 0.1
TRANSITION = np.array([[0.95, 0.05],
                       [0.05, 0.95]])


def systematic_resample(log_weights):
    A = log_weights.max()
    normalizer = np.log(np.exp(log_weights - A).sum()) + A
    weights = np.exp(log_weights - normalizer)
    ns = len(weights)
    cdf = np.cumsum(weights)
    cutoff = (np.random.rand() + np.arange(ns))/ns
    return np.digitize(cutoff, cdf)

def ESS(log_weights):
    A = log_weights.max()
    normalizer = np.log(np.exp(log_weights - A).sum()) + A
    log_normalized = 2*(log_weights - normalizer)
    B = log_normalized.max()
    log_denominator = np.log(np.sum(np.exp(log_normalized - B))) + B
    return np.exp(-log_denominator)


def run_smc_1D( Y, K, proposal, verbose=True):
    """ Run an SMC algorithm using K particles, and proposal distribution `proposal`,
        which returns a value and its proposal log probability.

        `factorial_hmm.baseline_proposal` samples from the transition dynamics.
        `factorial_hmm.make_nn_proposal` generates a proposal using a learned network. """

    T = len(Y)
    X_hat = np.zeros((K, T), dtype=int)
    ancestry = np.empty((K, T), dtype=int)
    ln_q = 0.0
    for t in range(1, len(Y)):
        if t ==0:
            X_hat[:, 0], ln_q = Normal(0,1)(1 * (np.random.rand(K, len(devices)) < P_INIT), Y[0])
            log_weights = stats.norm(Y[0], SIGMA).logpdf(np.dot(X_hat[:, 0], devices)) - ln_q
            ESS_history = np.empty((T,))
            ESS_history[0] = ESS(log_weights)
            if ESS_history[0] < K * RESAMPLE_THRESH:
                X_hat = X_hat[systematic_resample(log_weights)]
                log_weights[:] = 0.0  # np.log(np.mean(np.exp(log_weights)))
            ancestry[:, 0] = np.arange(K)
        else:
            X_hat[:, t], ln_q = proposal(X_hat[:, t - 1], Y[t])
            ln_p_trans = np.sum(np.log(TRANSITION[0, 0]) * (X_hat[:, t] == X_hat[:, t - 1]) +
                                np.log(TRANSITION[0, 1]) * (X_hat[:, t] != X_hat[:, t - 1]), 1)
            # assert np.isfinite(ln_q) #
            log_weights += stats.norm(Y[t], SIGMA).logpdf(np.dot(X_hat[:, t], devices)) + ln_p_trans - ln_q
            ESS_history[t] = ESS(log_weights)
            if ESS_history[t] < K * RESAMPLE_THRESH:
                if verbose:
                    print( "RESAMPLE t=%d,ESS=%f" %(t, ESS_history[t]))
                indices = systematic_resample(log_weights)
                X_hat = X_hat[indices]
                log_weights[:] = 0.0  # np.log(np.mean(np.exp(log_weights)))
                ancestry = ancestry[indices]
            ancestry[:, t] = np.arange(K)
    indices = systematic_resample(log_weights)
    ancestry = ancestry[indices]
    ancestry[:, -1] = np.arange(K)
    return X_hat[indices], ancestry, ESS_history