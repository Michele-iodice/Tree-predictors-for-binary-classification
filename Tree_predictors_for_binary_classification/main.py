import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from Tree_predictors_for_binary_classification.criterion.SplittingFunction import gini_score, entropy_score, information_gain, mse_score
from Tree_predictors_for_binary_classification.criterion.StoppingFunction import max_depth_reached, min_samples_per_leaf, min_impurity_threshold
from Tree_predictors_for_binary_classification.TreeConstruction.TreePredictor import TreePredictor


if __name__ == '__main__':
    # hyper_parameter
    splitting_criteria = [gini_score, entropy_score, information_gain, mse_score]
    stopping_criterion = min_samples_per_leaf
    maxDepths=list(range(1, 21))
    min_samples= [2, 3, 4, 5, 10, 15, 20, 30, 50]
    impurity_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    column_names = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
        "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color", "population", "habitat"
    ]

    # Load the mushroom dataset version 2
    # data = pd.read_csv("../Tree_predictors_for_binary_classification/Dataset/MushroomDataset/secondary_data.csv",
    #                    delimiter=';')
    # Load the mushroom dataset version 3
    data = pd.read_csv('../Tree_predictors_for_binary_classification/Dataset/mushroom/mushroom.data',
                       header=None, names=column_names)

    # Preview the dataset
    print(data.head())
    print(data.info())

    target_column='class'
    X = data.drop(columns=target_column).values
    y = (data[target_column] == 'p').astype(int).values
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    best_val_error = 1
    best_hyper_parameter = None
    best_split_criteria = None
    for split_criterion in splitting_criteria:
        val_errors = []
        train_errors = []
        for param in min_samples:
            predictor = TreePredictor(splitting_criterion=split_criterion,
                                      stopping_criterion=stopping_criterion,
                                      stopping_param=param)

            predictor.fit(X_train, y_train)
            train_error = predictor.training_error(X_train, y_train)
            val_error = predictor.training_error(X_val, y_val)
            val_errors.append(val_error)
            train_errors.append(train_error)
            print(
                f"Split Criterion = {split_criterion.__name__}, Param = {param}: Train Error = {train_error:.2f}, "
                f"Validation error = {val_error:.2f}")

            if val_error < best_val_error:
                best_val_error = val_error
                best_hyper_parameter = param
                best_split_criteria = split_criterion

        plt.figure(figsize=(10, 6))
        plt.plot(min_samples, train_errors, label='Training Error', color='blue', marker='o', linestyle='-')
        plt.plot(min_samples, val_errors, label='Validation Error', color='red', marker='o', linestyle='-')
        plt.xlabel('Min samples par leaf')
        plt.ylabel('Error')
        plt.title(
            f'Training and Validation Error vs Min samples par leaf using split:{split_criterion.__name__},'
            f' and stopping:{stopping_criterion.__name__}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'../Tree_predictors_for_binary_classification/results/dataset_v3/min_samples/'
                    f'{split_criterion.__name__}_graphic.jpg')
        plt.show()

    # Final evaluation on the test set
    print(f"The best param value is: {best_hyper_parameter}, Validation error = {best_val_error:.2f}")
    best_predictor = TreePredictor(splitting_criterion=best_split_criteria,
                                   stopping_criterion=stopping_criterion,
                                   stopping_param=best_hyper_parameter)
    best_predictor.fit(X_train_val, y_train_val)
    test_accuracy = best_predictor.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

