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

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    features = np.asarray(features)
    labels = np.asarray(labels)
    
    best_score = 0.0
    best_features = None
    best_threshold = None

    for f_idx in feature_indices:
        col_values = features[:, f_idx]
        unique_val = np.unique(col_values)

        if len(unique_val) <=1:
            continue

        thresholds = (unique_val[:-1]+ unique_val[1:])/ 2.0

        for t in thresholds:
            _, left_y, _, right_y = split_dataset(features, labels, f_idx, t)

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            score =  split_score(labels, left_y, right_y)

            if score > best_score:
                best_score = score
                best_features = f_idx
                best_threshold = float(t)

    return {
        "feature_index" : best_features,
        "threshold": best_threshold,
        "score": float(best_score)
    }

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    labels = np.asarray(labels)

    if len(np.unique(labels)) <= 1:
        return True

    if max_depth is not None and depth >= max_depth:
        return True
    
    if len(labels) < min_samples_split :
        return True

    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    values, counts = np.unique(labels, return_counts=True)
    winner_idx = np.argmax(counts)
    return int(values[winner_idx])

# Step 7 - build_tree
import numpy as np

def build_tree(features, labels, depth=0, max_depth=None, min_samples_split=2, feature_subset=None):
    features, labels = np.asarray(features), np.asarray(labels)
    
    # 1. Base case: leaf
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {"leaf": True, "prediction": leaf_prediction(labels)}
    
    # 2. Pick candidates & find best split
    candidates = range(features.shape[1]) if feature_subset is None else feature_subset
    best = best_split(features, labels, candidates)
    
    # Can't improve? Leaf.
    if best["feature_index"] is None or best["score"] <= 0:
        return {"leaf": True, "prediction": leaf_prediction(labels)}
    
    # 3. Split & Recurse
    f_idx, thresh = best["feature_index"], best["threshold"]
    l_X, l_y, r_X, r_y = split_dataset(features, labels, f_idx, thresh)
    
    return {
        "leaf": False,
        "feature_index": int(f_idx),
        "threshold": float(thresh),
        "left": build_tree(l_X, l_y, depth + 1, max_depth, min_samples_split, feature_subset),
        "right": build_tree(r_X, r_y, depth + 1, max_depth, min_samples_split, feature_subset),
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    node = tree
    
    # Keep walking down as long as we haven't hit a leaf
    while not node["leaf"]:
        f_idx = node["feature_index"]
        threshold = node["threshold"]
        
        if example[f_idx] <= threshold:
            node = node["left"]
        else:
            node = node["right"]
            
    return int(node["prediction"])

# Step 9 - predict_tree
import numpy as np

def predict_tree(tree, features):
    features = np.asarray(features)
    
    predictions = [predict_example_tree(tree, row) for row in features]
    
    return np.array(predictions, dtype=int)

# Step 10 - bootstrap_sample
import numpy as np

def bootstrap_sample(features, labels, rng):
    features = np.asarray(features)
    labels = np.asarray(labels)
    
    n_samples = features.shape[0]
    
    sampled_indices = rng.choice(n_samples, size=n_samples, replace=True)
    
    sampled_features = features[sampled_indices]
    sampled_labels = labels[sampled_indices]
    
    return sampled_features, sampled_labels

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    return rng.choice(num_features, size=num_to_pick, replace=False)

# Step 12 - train_forest
import numpy as np

def train_forest(
    features,
    labels,
    num_trees,
    num_features_per_split=None,
    max_depth=None,
    min_samples_split=2,
    random_state=None,
    rng=None
):
    features = np.asarray(features)
    labels = np.asarray(labels)
    
    # Handle random_state or rng seamlessly
    if rng is not None:
        generator = rng
    elif isinstance(random_state, np.random.Generator):
        generator = random_state
    else:
        generator = np.random.default_rng(random_state)
        
    n_samples, d = features.shape
    
    # Default feature subset size: round(sqrt(d)) with a minimum of 1
    if num_features_per_split is None:
        num_to_pick = max(1, int(round(np.sqrt(d))))
    else:
        num_to_pick = num_features_per_split
        
    forest = []
    for _ in range(num_trees):
        # 1. Bootstrap sample of the training rows
        boot_X, boot_y = bootstrap_sample(features, labels, generator)
        
        # 2. Pick a random subset of feature column indices
        cols = feature_subset(d, num_to_pick, generator)
        
        # 3. Build a tree restricted to these features
        tree = build_tree(
            boot_X,
            boot_y,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            feature_subset=cols
        )
        
        forest.append({
            "tree": tree,
            "feature_indices": cols
        })
        
    return forest

# Step 13 - combine_predictions
import numpy as np

def combine_predictions(tree_predictions):
    tree_predictions = np.asarray(tree_predictions)
    
    # tree_predictions shape: (num_trees, n)
    n = tree_predictions.shape[1]
    final_preds = []
    
    for j in range(n):
        col_votes = tree_predictions[:, j]
        # Re-use the majority voting logic (or call leaf_prediction directly)
        values, counts = np.unique(col_votes, return_counts=True)
        winner_class = values[np.argmax(counts)]
        final_preds.append(winner_class)
        
    return np.array(final_preds, dtype=int)

# Step 14 - predict_forest
import numpy as np

def predict_forest(forest, features):
    features = np.asarray(features)
    
    tree_predictions = [predict_tree(entry["tree"], features) for entry in forest]
    
    tree_predictions = np.vstack(tree_predictions)
    
    return combine_predictions(tree_predictions)

# Step 15 - accuracy (not yet solved)
# TODO: implement

