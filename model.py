"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np

def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)

    probabilities = counts / len(labels)

    gini = 1.0 - np.sum(probabilities ** 2)

    return float(gini)

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    features = np.asarray(features)
    labels = np.asarray(labels)

    left_mask = features[:, feature_index] <=threshold
    right_mask = ~left_mask

    left_features = features[left_mask]
    left_labels = labels[left_mask]

    right_features = features[right_mask]
    right_labels = labels[right_mask]

    return left_features, left_labels, right_features, right_labels

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    n_parent = len(parent_labels)
    if n_parent == 0:
        return 0.0

    n_left = len(left_labels)
    n_right = len(right_labels)

    if n_left == 0 or n_right == 0:
        return 0.0

    parent_imp = impurity(parent_labels)

    left_imp = impurity(left_labels)
    right_imp = impurity(right_labels)

    weighted_child_imp = (n_left/n_parent) * left_imp + (n_right/n_parent) * right_imp

    return float(parent_imp - weighted_child_imp)

# Step 4 - best_split (not yet solved)
# TODO: implement

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

