import pandas as pd
import matplotlib.pyplot as plt
from numpy import NaN
from sklearn.model_selection import train_test_split
from Tree_predictors_for_binary_classification.criterion.SplittingFunction import gini_score, entropy_score, information_gain, mse_score
from Tree_predictors_for_binary_classification.criterion.StoppingFunction import max_depth_reached, min_samples_per_leaf, min_impurity_threshold
from Tree_predictors_for_binary_classification.TreeConstruction.TreePredictor import TreePredictor
import numpy as np
from sklearn.model_selection import KFold

if __name__ == '__main__':
    # hyper_parameter
    splitting_criteria = gini_score
    stopping_criteria = max_depth_reached
    maxDepth = 5
    column_names = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
        "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color", "population", "habitat"
    ]

    data = pd.read_csv('../Dataset/mushroom/mushroom.data',
                       header=None, names=column_names)
    data.replace("?", NaN, inplace=True)

    # Preview the dataset
    print(data.head())
    print(data.info())

    target_column='class'
    X = data.drop(columns=target_column).values
    y = (data[target_column] == 'p').astype(int).values
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_val_errors = []
    fold_train_errors = []
    fold = 1
    for train_index, val_index in kf.split(X_train_val):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        predictor = TreePredictor(splitting_criterion=splitting_criteria,
                                  stopping_criterion=stopping_criteria,
                                  stopping_param=maxDepth)

        predictor.fit(X_train, y_train)
        train_error = predictor.training_error(X_train, y_train)
        val_error = predictor.training_error(X_val, y_val)
        fold_train_errors.append(train_error)
        fold_val_errors.append(val_error)
        print(f"Fold {fold}: Train Error = {train_error:.2f}, Validation Error = {val_error:.2f}")
        fold = fold + 1

    avg_train_error = np.mean(fold_train_errors)
    avg_val_error = np.mean(fold_val_errors)

    print(f"Train Error = {avg_train_error:.2f}, Validation error = {avg_val_error:.2f}")

    test_predictor = TreePredictor(splitting_criterion=splitting_criteria,
                                   stopping_criterion=stopping_criteria,
                                   stopping_param=maxDepth)
    test_predictor.fit(X_train_val, y_train_val)
    test_accuracy = test_predictor.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, fold)), fold_train_errors, label='Training Error', color='blue', marker='o', linestyle='-')
    plt.plot(list(range(1, fold)), fold_val_errors, label='Validation Error', color='red', marker='o', linestyle='-')
    plt.xlabel('Fold')
    plt.ylabel('Error')
    plt.title(f'Train and Validation error report')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../results/dataset_v3/train_result/'
                f'{splitting_criteria.__name__}AND{stopping_criteria.__name__}train_graphic.jpg')
    plt.show()
