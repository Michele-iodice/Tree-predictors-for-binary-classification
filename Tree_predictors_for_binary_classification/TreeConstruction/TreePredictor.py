import numpy as np
import pandas as pd
from Tree_predictors_for_binary_classification.TreeConstruction.TreeNode import TreeNode
from Tree_predictors_for_binary_classification.criterion.SplittingFunction import entropy


class TreePredictor:
    def __init__(self, splitting_criterion, stopping_criterion, stopping_param,
                 maxDepth=980, minSamplesParLeaf=2, minImpurityThreshold=0.1):
        """
        Initializes the TreePredictor.

        Parameters:
        - splitting_criterion: A function that computes the splitting criterion score.
        - stopping_criterion: A function that checks whether to stop tree growth.
        """
        self.root = None  # Root node of the tree
        self.splitting_criterion = splitting_criterion  # Function for splitting
        self.stopping_criterion = stopping_criterion  # Function for stopping
        self.stopping_param = stopping_param  # param for stopping Function
        self.maxDepth = maxDepth
        self.minSamplesParLeaf = minSamplesParLeaf
        self.minImpurityThreshold = minImpurityThreshold

    def fit(self, X, y):
        """
        Trains the tree predictor on the given training set.

        Parameters:
        - X: Features of the training set.
        - y: Labels of the training set.
        """
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        """
        Recursively grows the decision tree.

        Parameters:
        - X: Features of the training set.
        - y: Labels of the training set.
        - depth: Current depth of the tree.

        Returns:
        - A TreeNode representing the root of the subtree.
        """
        # Check stopping criteria
        if self.stopping_criterion(y, depth, self.stopping_param) or len(np.unique(y)) == 1:
            return TreeNode(is_leaf=True, class_label=self._most_common_class(y) if len(y) > 0 else None)

        if depth >= self.maxDepth or len(y) <= self.minSamplesParLeaf or entropy(y) < self.minImpurityThreshold:
            return TreeNode(is_leaf=True, class_label=self._most_common_class(y) if len(y) > 0 else None)

        # Determine the best split
        best_criteria= self._best_split(X, y)

        if best_criteria is None:  # No valid split found
            return TreeNode(is_leaf=True, class_label=self._most_common_class(y))

        left_indices, right_indices = self._split_data(X, y, best_criteria)
        if len(left_indices) <= 1 or len(right_indices) <= 1:
            return TreeNode(is_leaf=True, class_label=self._most_common_class(y) if len(y) > 0 else None)

        # Recursively grow the left and right subtrees
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return TreeNode(is_leaf=False,
                        decision_function=best_criteria,
                        left=left_child,
                        right=right_child
                        )

    def _best_split(self, X, y):
        """
        Finds the best split for the data.

        Parameters:
        - X: Features of the training set.
        - y: Labels of the training set.

        Returns:
        - The criteria for the best split.
        """
        # Implement your logic to find the best split based on self.splitting_criterion
        best_score = float('inf')
        best_criteria = None

        for feature_index in range(X.shape[1]):
            feature_values = X[:, feature_index]

            if np.issubdtype(feature_values.dtype, np.number):  # Numerical feature
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    left_indices = (feature_values <= threshold) & (~np.isnan(feature_values))
                    right_indices = (feature_values > threshold) & (~np.isnan(feature_values))

                    if np.any(left_indices) and np.any(right_indices):
                        left_y = y[left_indices]
                        right_y = y[right_indices]
                        score = self.splitting_criterion(left_y, right_y)

                        if score < best_score:
                            best_score = score
                            best_criteria = lambda x: x[feature_index] <= threshold

            else:  # Categorical feature
                unique_values = pd.unique(feature_values)

                for category in unique_values:
                    left_indices = (feature_values == category) & (~pd.isna(feature_values))
                    right_indices = (feature_values != category) & (~pd.isna(feature_values))

                    if np.any(left_indices) and np.any(right_indices):
                        left_y = y[left_indices]
                        right_y = y[right_indices]
                        score = self.splitting_criterion(left_y, right_y)

                        if score < best_score:
                            best_score = score
                            best_criteria = lambda x: x[feature_index] == category

        return best_criteria

    def _split_data(self, X, y, criteria):
        """
        Splits the dataset based on the criteria.

        Parameters:
        - X: Features of the dataset.
        - y: Labels of the dataset.
        - criteria: The criteria to split the data.

        Returns:
        - Indices of the left and right splits.
        """
        left_indices = np.array([criteria(X[i]) for i in range(len(X))])
        right_indices = ~left_indices
        return left_indices, right_indices

    def _most_common_class(self, y):
        """
        Determines the most common class label.

        Parameters:
        - y: Labels of the dataset.

        Returns:
        - The most common class label.
        """
        return np.bincount(y).argmax()

    def predict(self, X):
        """
        Makes predictions for the given input data.

        Parameters:
        - X: Features for which to make predictions.

        Returns:
        - Predicted class labels.
        """
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _predict_sample(self, sample, node):
        """
        Predicts the class label for a single sample.

        Parameters:
        - sample: A single sample to predict.
        - node: The current node in the decision tree.

        Returns:
        - Predicted class label.
        """
        if node.is_leaf:
            return node.class_label

        # Determine the child node to follow
        if node.decision_function(sample):
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

    def evaluate(self, X, y):
        """
        Evaluates the accuracy of the model on a given dataset.

        Parameters:
        - X: Features of the evaluation dataset.
        - y: Labels of the evaluation dataset.

        Returns:
        - Accuracy of the model.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def training_error(self, X, y):
        """
        Computes the training error according to 0-1 loss.

        Parameters:
        - X: Features of the training dataset.
        - y: True labels of the training dataset.

        Returns:
        - Training error rate.
        """
        predictions = self.predict(X)
        incorrect_predictions = np.sum(predictions != y)
        return incorrect_predictions / len(y)
