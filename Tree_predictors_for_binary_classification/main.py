import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Tree_predictors_for_binary_classification.HyperparameterTuning import hyperParameterTuning, fix_hyperParameterTuning
from Tree_predictors_for_binary_classification.criterion.SplittingFunction import gini_score, entropy_score, information_gain, mse_score
from Tree_predictors_for_binary_classification.criterion.StoppingFunction import max_depth_reached, min_samples_per_leaf, min_impurity_threshold
from Tree_predictors_for_binary_classification.TreeConstruction.TreePredictor import TreePredictor


if __name__ == '__main__':
    # hyper_parameter
    path= '../Tree_predictors_for_binary_classification/results/dataset_v3/k_fold/allParameter_tuning/'
    splitting_criteria = [gini_score, entropy_score, information_gain, mse_score]
    stopping_criteria = [min_samples_per_leaf, max_depth_reached,  min_impurity_threshold]
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

    # tuning all the hyper-parameter
    best_val_error, best_hyper_parameter, best_split_criteria, best_stop_criteria=hyperParameterTuning(
       X, y, splitting_criteria, stopping_criteria, maxDepths, min_samples, impurity_threshold, path)

    # tuning with a fixed splitting and stopping function
    # best_val_error, best_hyper_parameter, best_split_criteria, best_stop_criteria = fix_hyperParameterTuning(
    #   X, y, gini_score, max_depth_reached, maxDepths, path)

    # Final evaluation on the test set
    best_predictor = TreePredictor(splitting_criterion=best_split_criteria,
                                   stopping_criterion=best_stop_criteria,
                                   stopping_param=best_hyper_parameter)
    best_predictor.fit(X_train_val, y_train_val)
    test_accuracy = best_predictor.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    plt.figure(figsize=(8, 5))
    plt.bar(['Validation Error', 'Test Accuracy'], [best_val_error, test_accuracy], color=['red', 'green'])
    plt.ylabel('Error/Accuracy')
    plt.title(f'Best Validation Error vs Test Accuracy (Param = {best_hyper_parameter})')
    plt.savefig(path+'test_report.jpg')
    plt.show()
