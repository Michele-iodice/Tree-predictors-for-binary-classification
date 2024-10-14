from Tree_predictors_for_binary_classification.criterion.SplittingFunction import entropy_score


def max_depth_reached(depth, max_depth):
    """Check if maximum depth has been reached."""
    return depth >= max_depth


def min_samples_per_leaf(y, min_samples_leaf):
    """Check if the leaf has a minimum number of samples."""
    return len(y) < min_samples_leaf


def min_impurity_threshold(X, y, depth, impurity_threshold=0.1):
    """
    Stopping criterion based on minimum impurity threshold.

    Parameters:
    - X: Features of the dataset.
    - y: Labels of the dataset.
    - depth: Current depth of the tree.
    - impurity_threshold: The impurity threshold for stopping.

    Returns:
    - True if stopping criterion is met, otherwise False.
    """
    current_impurity = entropy_score(y, y)  # Or use any impurity measure
    return current_impurity < impurity_threshold