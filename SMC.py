import numpy as np

var = 1
d = 20
N = 2000
K = 35
a = np.random.uniform(size=d)
b = np.random.uniform(size=(d, d))
b = b @ np.transpose(b)
inv_cov = np.linalg.inv(b)
phi = []
phi.append(5e-3)


def get_prob(proposal, X, cov):
    d = np.zeros(N)
    for k in range(N):
        A = - 0.5 * ((proposal[k, :] - a) @ cov @ np.transpose(proposal[k, :] - a))
        D = - 0.5 * ((X[k, :] - a) @ cov @ np.transpose(X[k, :] - a))
        d[k] = A - D
    return d


def get_weights(X, a, delta, inv_cov):
    vec = (X - a);
    lw = - delta * 0.5 * vec @ inv_cov @ vec.T
    # A = - 0.5 * (np.sum((X - a) @ next_cov * (X - a), axis = 1))
    # D = - 0.5 * (np.sum((X - a) @ curr_cov * (X - a), axis = 1))
    # W = A - D
    lw -= np.max(lw)
    return np.exp(lw)


# Samples initialization
X = np.random.multivariate_normal(a, b / phi[0], size=N)
W = np.full(N, 1 / N)

for i in range(K):
    # Resample
    I = np.random.choice(np.arange(N), size=N, p=W)
    X = X[I, :]
    W = np.full(N, 1 / N)

    # MCMC step
    accep = np.zeros(N)
    for j in range(10):

        if j > 3:
            var = 2 * np.log((i + 2) / 2) * b / phi[i] / (i + 1)
        else:
            var = 4 * b / phi[i]
        if i == 0:
            var = var * 1000000
        R = np.random.multivariate_normal(np.zeros(d), var, size=N)
        proposal = X + R
        U = np.log(np.random.uniform(size=N))
        alpha = get_prob(proposal, X, phi[i] * inv_cov)
        # print(alpha)
        for k in range(N):
            if U[k] < alpha[k]:
                accep[k] += 1
                X[k, :] = proposal[k, :]

    # Computing weights
    accep = np.sum(accep) / N
    accep = accep / 10
    print(accep)
    # sample_mean = np.sum(X, axis = 0)/N

    # print("Sample Mean: ", sample_mean[:5])
    phi.append(phi[i] + (1 - phi[0]) / K)
    delta = phi[i + 1] - phi[i]
    for p in range(N):
        W[p] = get_weights(X[p, :], a, delta, inv_cov)
    # print(W)
    W = W / np.sum(W)

print(phi)
sample_mean = W.T @ X
sample_cov = np.zeros((d, d))
for i in range(N):
    vec = X[i, :] - a.reshape(-1, 1)
    sample_cov += W[i] * (vec.T @ vec)
sample_cov = sample_cov / N
print("True Mean: ", a[:5])
print("Sample Mean: ", sample_mean[:5])
print("True Covariance: ", b[:5, :5])
print("Sample Covariance: ", sample_cov[:5, :5])