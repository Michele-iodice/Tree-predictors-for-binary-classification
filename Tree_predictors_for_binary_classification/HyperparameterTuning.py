import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from Tree_predictors_for_binary_classification.TreeConstruction.TreePredictor import TreePredictor


def hyperParameterTuning(X, y, splitting_criteria, stopping_criteria, maxDepths, min_samples, impurity_threshold, path):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_val_error = 1
    best_hyper_parameter = None
    best_split_criteria = None
    best_stop_criteria = None
    for split_criterion in splitting_criteria:
        for stopping_criterion in stopping_criteria:
            val_errors = []
            train_errors = []
            params = maxDepths
            params_name="max depths"

            if stopping_criterion.__name__ == "min_samples_per_leaf":
                params=min_samples
                params_name = "min samples"
            if stopping_criterion.__name__ == "min_impurity_threshold":
                params=impurity_threshold
                params_name = "impurity threshold"

            for param in params:
                fold_val_errors = []
                fold_train_errors = []
                fold=1
                for train_index, val_index in kf.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]
                    predictor = TreePredictor(splitting_criterion=split_criterion,
                                              stopping_criterion=stopping_criterion,
                                              stopping_param=param)

                    predictor.fit(X_train, y_train)
                    train_error = predictor.training_error(X_train, y_train)
                    val_error = predictor.training_error(X_val, y_val)
                    fold_train_errors.append(train_error)
                    fold_val_errors.append(val_error)
                    print(f"Fold {fold}: Train Error = {train_error:.2f}, Validation Error = {val_error:.2f}")
                    fold=fold+1

                avg_train_error = np.mean(fold_train_errors)
                avg_val_error = np.mean(fold_val_errors)
                val_errors.append(avg_val_error)
                train_errors.append(avg_train_error)
                print(f"Split Criterion = {split_criterion.__name__},Stop Criterion = {stopping_criterion.__name__},"
                      f" Param = {param}:\nTrain Error = {avg_train_error:.2f}, Validation error = {avg_val_error:.2f}")

                if avg_val_error < best_val_error:
                    best_val_error = avg_val_error
                    best_hyper_parameter = param
                    best_split_criteria = split_criterion
                    best_stop_criteria = stopping_criterion
            print(fold_train_errors)

            plt.figure(figsize=(10, 6))
            print(train_errors)
            plt.plot(params, train_errors, label='Training Error', color='blue', marker='o', linestyle='-')
            plt.plot(params, val_errors, label='Validation Error', color='red', marker='o', linestyle='-')
            plt.xlabel(f'{params_name}')
            plt.ylabel('Error')
            plt.title(
                f'Training and Validation Error using as splitting:{split_criterion.__name__},'
                f' and stopping:{stopping_criterion.__name__}')
            plt.legend()
            plt.grid(True)
            fig_path = path + f'{split_criterion.__name__}AND{stopping_criterion.__name__}_graphic.jpg'
            plt.savefig(fig_path)
            plt.show()

    print(f"The best splitting and stopping function are split: {best_split_criteria.__name__}, "
          f"Stopping= {best_stop_criteria.__name__}")
    print(f"The best param value is: {best_hyper_parameter}, Validation error = {best_val_error:.2f}")

    return best_val_error, best_hyper_parameter, best_split_criteria, best_stop_criteria


def fix_hyperParameterTuning(X, y, splitting_criteria, stopping_criteria, params, path):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_val_error = 1
    best_hyper_parameter = None
    val_errors = []
    train_errors = []
    params_name = "max depths"
    if stopping_criteria.__name__ == "min_samples_per_leaf":
        params_name = "min samples"
    if stopping_criteria.__name__ == "min_impurity_threshold":
        params_name = "impurity threshold"

    for param in params:
        fold_val_errors = []
        fold_train_errors = []
        fold = 1
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            predictor = TreePredictor(splitting_criterion=splitting_criteria,
                                      stopping_criterion=stopping_criteria,
                                      stopping_param=param)

            predictor.fit(X_train, y_train)
            train_error = predictor.training_error(X_train, y_train)
            val_error = predictor.training_error(X_val, y_val)
            fold_train_errors.append(train_error)
            fold_val_errors.append(val_error)
            print(f"Fold {fold}: Train Error = {train_error:.2f}, Validation Error = {val_error:.2f}")
            fold = fold + 1

        avg_train_error = np.mean(fold_train_errors)
        avg_val_error = np.mean(fold_val_errors)
        val_errors.append(avg_val_error)
        train_errors.append(avg_train_error)
        print(f"Param = {param}: Train Error = {avg_train_error:.2f}, Validation error = {avg_val_error:.2f}")

        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            best_hyper_parameter = param

    plt.figure(figsize=(10, 6))
    plt.plot(params, train_errors, label='Training Error', color='blue', marker='o', linestyle='-')
    plt.plot(params, val_errors, label='Validation Error', color='red', marker='o', linestyle='-')
    plt.xlabel(f'{params_name}')
    plt.ylabel('Error')
    plt.title(
        f'Training and Validation Error using as splitting:{splitting_criteria.__name__},'
        f' and stopping:{stopping_criteria.__name__}')
    plt.legend()
    plt.grid(True)
    fig_path=path+f'{splitting_criteria.__name__}AND{stopping_criteria.__name__}_graphic.jpg'
    plt.savefig(fig_path)
    plt.show()

    print(f"The best param value is: {best_hyper_parameter}, Validation error = {best_val_error:.2f}")

    return best_val_error, best_hyper_parameter, splitting_criteria, stopping_criteria
