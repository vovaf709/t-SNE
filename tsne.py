import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class TSNE:
    """Naive implementation of t-SNE"""

    def __init__(
        self,
        n_components=2,
        *,
        perplexity=30.0,
        early_exaggeration=12.0,
        n_iter=1000,
        random_state=None,
        eps=1e-1,
        lr=200,
        warmup_steps=250,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.n_iter = n_iter
        self.lr = lr
        self.eps = eps
        self.warmup_steps = warmup_steps

        np.random.seed(random_state)

    @staticmethod
    def _perplexity(p, eps=1e-10):
        entropy = (-p * np.log2(p + 1e-10)).sum(axis=1)

        return 2 ** entropy

    @staticmethod
    def _conditional_probs(X, sigmas2):
        unnormalized = np.exp(-pairwise_distances(X) ** 2 / (2 * sigmas2.reshape((-1, 1))))
        np.fill_diagonal(unnormalized, 0)

        return unnormalized / unnormalized.sum(axis=1, keepdims=True)

    @staticmethod
    def _q(X_lowdim):
        unnormalized = 1 / (1 + pairwise_distances(X_lowdim) ** 2)
        np.fill_diagonal(unnormalized, 0)

        return unnormalized / unnormalized.sum()

    def _calc_sigmas2(self, X):
        print('Calculating variances...')

        progress_threshold = 0.25

        sigmas2 = np.ones(X.shape[0])
        ready_mask = np.zeros_like(sigmas2)

        while not np.prod(ready_mask):
            readiness = np.mean(ready_mask)

            if readiness > progress_threshold:
                print(f'{int(100 * progress_threshold)}%')
                progress_threshold += 0.25

            probs = self._conditional_probs(X, sigmas2)

            for i, perplexity in enumerate(self._perplexity(probs)):
                if perplexity > self.perplexity + self.eps:
                    sigmas2[i] /= 2
                elif perplexity < self.perplexity - self.eps:
                    sigmas2[i] += sigmas2[i] / 2
                else:
                    ready_mask[i] = 1

        print('100%')
        print('Variances are calculated!\n')

        return sigmas2

    def fit(self, X, y=None):
        sigmas2 = self._calc_sigmas2(X)
        cond_probs = self._conditional_probs(X, sigmas2)

        p = (cond_probs + cond_probs.T) / (2 * X.shape[0])

        X_lowdim = np.random.randn(X.shape[0], self.n_components) / 100
        prev_X_lowdim = np.copy(X_lowdim)

        print('Minimizing t-SNE objective...')
        for i in tqdm(range(self.n_iter)):

            q = self._q(X_lowdim)

            unnormalized = 1 / (1 + pairwise_distances(X_lowdim) ** 2)
            np.fill_diagonal(unnormalized, 0)

            grad = 4 * (
                (p * (1 + (self.early_exaggeration - 1) * (i < self.warmup_steps)) - q)[..., None]
                * (X_lowdim[:, None] - X_lowdim[None])
                * unnormalized[..., None]
            ).sum(axis=1)

            new_X_lowdim = (
                X_lowdim - self.lr * grad - (0.5 if i < self.warmup_steps else 0.8) * (X_lowdim - prev_X_lowdim)
            )

            prev_X_lowdim = X_lowdim
            X_lowdim = new_X_lowdim

        print('All done!\n')
        self._X_lowdim = X_lowdim

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        return self._X_lowdim
