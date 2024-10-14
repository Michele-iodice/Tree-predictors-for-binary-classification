import numpy as np


def gini_index(y):
    """Calculate the Gini index for a set of labels."""
    classes, counts = np.unique(y, return_counts=True)
    total_samples = counts.sum()
    gini = 1 - sum((count / total_samples) ** 2 for count in counts)
    return gini


def gini_score(y_left, y_right):
    """Calculate the Gini score for a split."""
    total_samples = len(y_left) + len(y_right)
    if total_samples == 0:
        return float('inf')  # Avoid division by zero
    return (len(y_left) / total_samples) * gini_index(y_left) + \
           (len(y_right) / total_samples) * gini_index(y_right)


def entropy(y):
    """Calculate the entropy for a set of labels."""
    classes, counts = np.unique(y, return_counts=True)
    total_samples = counts.sum()
    probabilities = counts / total_samples
    entropy_value = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy_value


def entropy_score(y_left, y_right):
    """Calculate the entropy score for a split."""
    total_samples = len(y_left) + len(y_right)
    if total_samples == 0:
        return float('inf')  # Avoid division by zero
    return (len(y_left) / total_samples) * entropy(y_left) + (len(y_right) / total_samples) * entropy(y_right)


def information_gain(y_left, y_right):
    """Calculate the Information Gain from a split."""
    y= np.concatenate(y_left, y_right)
    return entropy(y) - (len(y_left) / len(y)) * entropy(y_left) - (len(y_right) / len(y)) * entropy(y_right)


def mean_squared_error(y):
    """Calculate the Mean Squared Error for a set of labels."""
    if len(y) == 0:
        return float('inf')
    mean = np.mean(y)
    mse = np.mean((y - mean) ** 2)
    return mse


def mse_score(y_left, y_right):
    """Calculate the MSE score for a split.
     use it in regression trees to minimize the squared error
    """
    total_samples = len(y_left) + len(y_right)
    if total_samples == 0:
        return float('inf')  # Avoid division by zero
    return (len(y_left) / total_samples) * mean_squared_error(y_left) + \
           (len(y_right) / total_samples) * mean_squared_error(y_right)
