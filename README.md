# Tree predictors for binary classification
This repository presents the implementation of decision tree for binary
classification applied on the Mushroom Dataset to determine whether
mushrooms are poisonous. The tree predictors use single-feature bi-
nary tests as the decision criteria at any internal node. It was per-
formed hyperparameter tuning on the tree predictors. By reusing the
already implemented tree predictor class/structure a Random Forest
classifiers was implemented. The implementation is carried out using a
custom ‘TreePredictor‘ class, with different splitting and stopping cri-
teria. We further explore the effect of various hyperparameters, such
as ‘max depth‘, on model performance. The paper also discusses tech-
niques for efficient hyperparameter tuning, the results of the training
and test of the models and the importance of model evaluation on a
validation set

# file
- The python code is reported in the fold named: Tree_predictors_for_binary_classification
- the report on the results obtained is contained in the fold named: Report

# Reproducibility of the experiments
- In the Tree_predictors_for_binary_classification/TreeConstruction fold run the train.py file 
  to reproduce the training of a single Tree Predictor. 
  Remember to change the hyperparameter citated by the key-word # hyper_parameter based on what you want to train.
- In the Tree_predictors_for_binary_classification fold run the main.py file
  to reproduce the hyperparameter tuning of the Tree Predictor.
- In the Tree_predictors_for_binary_classification/RandomForest fold run the RandomForest.py file
  to reproduce the training and hyperparameter tuning of the Random Forest.
  Remember to change the hyperparameter based on what you want to do.

# citation
Author: Iodice Michele Attilio, 
email: micheleattilio.iodice@studenti.unimi.it,
Università degli Studi di Milano, year: 2024
