import numpy as np
from scipy.stats import iqr
from numpy.random import permutation

def mutualinformationx(x, y, fd_bins=None, permtest=False):
    """
    Compute mutual information between two vectors

    :param x: data vector
    :param y: data vector
    :param fd_bins: number of bins to use for distribution discretization (optional)
    :param permtest: perform permutation test and return mi in standard-Z values (optional)
    :return: (mi, entropy, fd_bins) tuple
    """
    # Ensure x and y are numpy arrays and have the same length
    x, y = np.asarray(x).flatten(), np.asarray(y).flatten()
    if len(x) != len(y):
        raise ValueError('X and Y must have equal length')

    # Determine the optimal number of bins for each variable
    if fd_bins is None:
        n_x = len(x)
        range_x = np.max(x) - np.min(x)
        fd_bins_x = np.ceil(range_x / (2.0 * iqr(x) * n_x ** (-1 / 3)))

        n_y = len(y)
        range_y = np.max(y) - np.min(y)
        fd_bins_y = np.ceil(range_y / (2.0 * iqr(y) * n_y ** (-1 / 3)))

        # Use the average
        fd_bins = np.ceil((fd_bins_x + fd_bins_y) / 2)

    # Bin data
    bins_x = np.digitize(x, np.linspace(np.min(x), np.max(x), int(fd_bins) + 1))
    bins_y = np.digitize(y, np.linspace(np.min(y), np.max(y), int(fd_bins) + 1))

    # Compute entropies
    hdat1 = np.histogram(x, bins=int(fd_bins))[0] / len(x)
    hdat2 = np.histogram(y, bins=int(fd_bins))[0] / len(y)

    entropy = [-np.sum(h * np.log2(h + np.finfo(float).eps)) for h in [hdat1, hdat2]]

    # Compute joint probabilities
    jointprobs, _, _ = np.histogram2d(x, y, bins=int(fd_bins))
    jointprobs /= jointprobs.sum()
    entropy.append(-np.sum(jointprobs * np.log2(jointprobs + np.finfo(float).eps)))

    # Mutual information
    mi = sum(entropy[:2]) - entropy[2]

    # Optional permutation testing
    if permtest:
        npermutes = 500
        perm_mi = np.zeros(npermutes)

        for permi in range(npermutes):
            shuffled_bins_y = permutation(bins_y)

            jointprobs, _, _ = np.histogram2d(x, shuffled_bins_y, bins=int(fd_bins))
            jointprobs /= jointprobs.sum()

            perm_jentropy = -np.sum(jointprobs * np.log2(jointprobs + np.finfo(float).eps))

            # Mutual information
            perm_mi[permi] = sum(entropy[:2]) - perm_jentropy

        mi = (mi - np.mean(perm_mi)) / np.std(perm_mi)

    return mi, entropy, fd_bins