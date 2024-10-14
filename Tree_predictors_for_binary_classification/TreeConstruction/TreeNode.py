class TreeNode:
    def __init__(self, is_leaf=False, class_label=None, decision_function=None, left=None, right=None):
        """
        Constructor to initialize the node.

        Parameters:
        - is_leaf (bool): Flag to check if the node is a leaf.
        - class_label (str): The class label for the leaf (if the node is a leaf).
        - decision_function (function): Function to perform the decision test (returns True/False).
        - left (TreeNode): Left child node (if any).
        - right (TreeNode): Right child node (if any).
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function

        if is_leaf and class_label is not None:
            self.set_leaf(class_label)
        else:
            self.is_leaf = False
            self.class_label = None

    def set_leaf(self, class_label):
        """
        Set the current node as a leaf with a specified class label.

        Parameters:
        - class_label (str): The class label to assign to this leaf (e.g., 'p' or 'e').
        """
        self.is_leaf = True
        self.class_label = class_label

    def predict(self, x):
        """
        Predict the class for a given data point by traversing the tree.

        Parameters:
        - x (numpy array): Input data point.

        Returns:
        - class_label (str): The predicted class label ('p' or 'e').
        """
        if self.is_leaf:
            return self.class_label
        else:
            # Use the decision function to decide the next child (left or right)
            if self.decision_function(x):
                return self.left.predict(x)
            else:
                return self.right.predict(x)
