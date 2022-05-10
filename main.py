from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS, ListedColormap

from mnist import mnist
from tsne import TSNE


def parse_args():
    parser = ArgumentParser("""Apply t-SNE to a subset of the MNIST dataset""")

    parser.add_argument(
        'n',
        type=int,
        help='Size of the subset',
    )

    parser.add_argument('out_path', type=Path, help='Path to the output .jpg file')

    parser.add_argument('--seed', type=int, help='Random seed')

    parser.add_argument(
        '--n_iter',
        type=int,
        help='Number of t-SNE iterations',
        default=1000,
    )

    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=512,
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.out_path.suffix == '.jpg', 'Please specify .jpg file'

    np.random.seed(args.seed)
    X_train, y_train, X_test, y_test = mnist()

    indices = np.random.permutation(X_train.shape[0])[: args.n]

    X_subset = X_train[indices]
    y_subset = y_train[indices]

    X_lowdim = TSNE(n_components=2, n_iter=args.n_iter, lr=args.lr, eps=1).fit_transform(X_subset)

    labels = y_subset.argmax(axis=1)
    colors = list(TABLEAU_COLORS.keys())

    plt.figure(figsize=(10, 10))
    for label, color in enumerate(colors):
        label_mask = labels == label
        plt.scatter(X_lowdim[label_mask, 0], X_lowdim[label_mask, 1], label=label, cmap=ListedColormap(colors))

    plt.legend()
    plt.savefig(args.out_path)
