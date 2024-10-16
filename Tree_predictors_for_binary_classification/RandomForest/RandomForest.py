import numpy as np
import pandas as pd
import math
import matplotlib as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from Tree_predictors_for_binary_classification.TreeConstruction.TreePredictor import TreePredictor
from Tree_predictors_for_binary_classification.criterion.SplittingFunction import gini_score
from Tree_predictors_for_binary_classification.criterion.StoppingFunction import min_samples_per_leaf


class RandomForest:
    def __init__(self, n_trees=100, max_features=None, min_samples_split=2):
        """
        Constructor to initialize the Random Forest.

        Parameters:
        - n_trees (int): Number of trees in the forest.
        - max_features (int): Number of features to consider when looking for the best split.
        - min_samples_split (int): Minimum number of samples required to split an internal node.
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        """
        Fit the Random Forest to the training data.

        Parameters:
        - X (numpy array or pd.DataFrame): Feature matrix.
        - y (numpy array or pd.Series): Target labels.
        """
        self.trees = []
        n_samples, n_features = X.shape
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                max_features = int(math.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(math.log2(n_features))
            else:
                raise ValueError(f"Unknown value for max_features: {self.max_features}")
        else:
            max_features = min(self.max_features*n_features, n_features)

        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            # If max_features is None, use all features
            if self.max_features is None:
                max_features = n_features
            else:
                max_features = min(max_features, n_features)

            # Randomly select features
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            X_sample = X_sample[:, feature_indices]

            # Create and train the decision tree
            tree = TreePredictor(splitting_criterion=gini_score,
                                 stopping_criterion=min_samples_per_leaf,
                                 stopping_param=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, feature_indices))  # Store the tree along with the feature indices used

    def predict(self, X):
        """
        Make predictions using the Random Forest.

        Parameters:
        - X (numpy array or pd.DataFrame): Feature matrix for prediction.

        Returns:
        - predictions (numpy array): Predicted class labels.
        """
        # Collect predictions from each tree
        predictions = np.zeros((X.shape[0], self.n_trees), dtype=object)

        for i, (tree, feature_indices) in enumerate(self.trees):
            # Only use the features that the tree was trained on
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)

        # Majority voting
        return np.array([np.bincount(pred).argmax() for pred in predictions.T])


def grid_search_random_forest(X_train, y_train, X_val, y_val, params):
    results=[]
    best_params = None
    best_score = np.inf  # or -np.inf if you're maximizing accuracy, AUC, etc.

    # Loop over the hyperparameter combinations
    for n_trees in params['n_trees']:
        for max_features in params['max_features']:
            for min_samples_split in params['min_samples_split']:
                # Create a new RandomForest with current hyperparameters
                rf = RandomForest(n_trees=n_trees,
                                  max_features=max_features,
                                  min_samples_split=min_samples_split)

                # Train the random forest on the training data
                rf.fit(X_train, y_train)

                # Evaluate on the validation set
                validation_error = rf.training_error(X_val, y_val)  # 0-1 loss or another metric

                print(
                    f'Params: n_trees={n_trees}, max_features={max_features}, min_samples_split={min_samples_split}')
                print(f'Validation Error: {validation_error}')

                # Keep track of the best parameters (assuming you're minimizing error)
                if validation_error < best_score:
                    best_score = validation_error
                    best_params = {
                        'n_trees': n_trees,
                        'max_features': max_features,
                        'min_samples_split': min_samples_split
                    }
                    # Store results
                    results.append({
                        'n_trees': n_trees,
                        'max_features': max_features,
                        'min_samples_split': min_samples_split,
                        'validation_error': validation_error
                    })

    print(f'Best Hyperparameters: {best_params}')
    print(f'Best Validation Error: {best_score}')
    return best_params, results


if __name__ == "__main__":
    param_grid = {
        'n_trees': [100, 200, 500],
        'max_features': ['sqrt', 'log2', 0.5],  # 50% of features as another option
        'min_samples_split': [2, 5, 10, 20, 50, 100]
    }

    # Load the dataset
    column_names = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
        "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color", "population", "habitat"
    ]

    data = pd.read_csv('../Dataset/mushroom/mushroom.data',
                       header=None, names=column_names)

    # Preview the dataset
    print(data.head())
    print(data.info())

    # Split data into features (X) and labels (y)
    target_column = 'class'
    X = data.drop(columns=target_column).values
    y = (data[target_column] == 'p').astype(int).values

    # Split into train and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # Assuming you have already split your dataset into X_train, y_train, X_val, y_val
    best_hyperparameters, result = grid_search_random_forest(X_train, y_train, X_val, y_val, param_grid)
    # Final model using the best hyperparameters
    final_rf = RandomForest(n_trees=best_hyperparameters['n_trees'],
                            max_features=best_hyperparameters['max_features'],
                            min_samples_split=best_hyperparameters['min_samples_split'])

    # Train the final model
    final_rf.fit(X_train, y_train)

    # Evaluate on the test set
    test_error = final_rf.training_error(X_test, y_test)  # Replace with actual evaluation method
    print(f'Test Error: {test_error}')
    # Plot results (for example, plot validation error for different n_trees)
    n_trees_values = [result['n_trees'] for result in result]
    validation_errors = [result['validation_error'] for result in result]

    plt.plot(n_trees_values, validation_errors, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Validation Error')
    plt.title('Random Forest Validation Error vs. Number of Trees')
    plt.show()
