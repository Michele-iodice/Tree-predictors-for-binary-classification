import pandas as pd
from sklearn.model_selection import train_test_split
from Tree_predictors_for_binary_classification.criterion.SplittingFunction import gini_score, entropy_score, information_gain, mse_score
from Tree_predictors_for_binary_classification.criterion.StoppingFunction import max_depth_reached, min_samples_per_leaf, min_impurity_threshold
from Tree_predictors_for_binary_classification.TreeConstruction.TreePredictor import TreePredictor

if __name__ == '__main__':
    splitting_criterion = gini_score
    # hyper_parameter
    maxDepths=[5]
    min_samples= 5
    impurity_threshold = 0.1

    # Load the mushroom dataset
    data = pd.read_csv("../Tree_predictors_for_binary_classification/Dataset/MushroomDataset/secondary_data.csv",
                       delimiter=';')

    # Preview the dataset
    print(data.head())
    print(data.info())

    target_column='class'
    X = data.drop(columns=target_column).values
    y = (data[target_column] == 'p').astype(int).values
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    best_train_error = 1
    best_hyper_parameter = None
    best_predictor= None
    for param in maxDepths:
        stopping_criterion = lambda X_stop, y_stop, depth: max_depth_reached(depth, param)
        # stopping_criterion = lambda X, y, depth: min_samples_per_leaf(y, param)
        #  stopping_criterion = lambda X, y, depth: min_impurity_threshold(X, y, depth,
        #                                                                  impurity_threshold=param)
        predictor = TreePredictor(splitting_criterion=splitting_criterion, stopping_criterion=stopping_criterion)

        predictor.fit(X_train, y_train)

        training_error = predictor.training_error(X_val, y_val)
        print(f" for the hyper param value: {param}")
        print(f"Training Error: {training_error:.2f}")

        if training_error < best_train_error:
            best_train_error = training_error
            best_hyper_parameter = param
            best_predictor=predictor

    # Final evaluation on the test set
    print(f"The best hyper param value is: {best_hyper_parameter}")
    test_accuracy = best_predictor.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

