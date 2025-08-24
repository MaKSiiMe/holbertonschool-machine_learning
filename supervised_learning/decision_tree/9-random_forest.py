#!/usr/bin/env python3
"""Module for building a decision tree"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Node:
    """Class representing a node in the decision tree"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def __str__(self):
        label = "root" if self.is_root else "node"
        root_line = (
            f"{label} [feature={self.feature}, threshold={self.threshold}]"
        )
        if not self.left_child and not self.right_child:
            return root_line
        parts = [root_line]
        if self.left_child:
            parts.append(
                left_child_add_prefix(str(self.left_child)).rstrip("\n")
            )
        if self.right_child:
            parts.append(
                right_child_add_prefix(str(self.right_child)).rstrip("\n")
            )
        return "\n".join(parts)

    def max_depth_below(self):
        """Calculate the maximum depth of the subtree rooted at this node"""
        if self.left_child is None and self.right_child is None:
            return self.depth
        depths = []
        if self.left_child is not None:
            depths.append(self.left_child.max_depth_below())
        if self.right_child is not None:
            depths.append(self.right_child.max_depth_below())
        return max(depths)

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes in the subtree rooted at this node"""
        if self.left_child is None and self.right_child is None:
            return 1
        if only_leaves:
            total = 0
            if self.left_child is not None:
                total += self.left_child.count_nodes_below(only_leaves=True)
            if self.right_child is not None:
                total += self.right_child.count_nodes_below(only_leaves=True)
            return total
        total = 1
        if self.left_child is not None:
            total += self.left_child.count_nodes_below(only_leaves=only_leaves)
        if self.right_child is not None:
            total += self.right_child.count_nodes_below(
                only_leaves=only_leaves
            )
        return total

    def get_leaves_below(self):
        """Get all leaves below this node (inclusive)"""
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Update the bounds of the subtree rooted at this node"""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in (self.left_child, self.right_child):
            if child is None:
                continue
            child.lower = dict(self.lower)
            child.upper = dict(self.upper)
            if self.feature is not None and self.threshold is not None:
                if child is self.left_child:
                    prev_lower = child.lower.get(self.feature, -np.inf)
                    child.lower[self.feature] = max(prev_lower, self.threshold)
                else:
                    prev_upper = child.upper.get(self.feature, np.inf)
                    child.upper[self.feature] = min(prev_upper, self.threshold)

        for child in (self.left_child, self.right_child):
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """Build an indicator function for points belonging to the node."""
        def is_large_enough(x):
            if (not hasattr(self, 'lower')
                    or len(getattr(self, 'lower', {})) == 0):
                return np.ones(x.shape[0], dtype=bool)
            conds = [x[:, k] > self.lower[k] for k in self.lower]
            return (
                np.all(np.stack(conds, axis=0), axis=0)
                if conds else
                np.ones(x.shape[0], bool)
            )

        def is_small_enough(x):
            if (not hasattr(self, 'upper')
                    or len(getattr(self, 'upper', {})) == 0):
                return np.ones(x.shape[0], dtype=bool)
            conds = [x[:, k] <= self.upper[k] for k in self.upper]
            return (
                np.all(np.stack(conds, axis=0), axis=0)
                if conds else
                np.ones(x.shape[0], bool)
            )

        self.indicator = lambda x: np.all(
            np.stack([is_large_enough(x), is_small_enough(x)], axis=0), axis=0
        )

    def pred(self, x):
        """Make a prediction for a single input sample."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Class representing a leaf node in the decision tree"""
    def __init__(self, value, depth=0, is_root=False):
        self.value = value
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = True
        self.sub_population = None

    def __str__(self):
        return f"-> leaf [value={self.value}]"

    def max_depth_below(self):
        """Calculate the maximum depth of the subtree rooted at this leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes in the subtree rooted at this leaf"""
        return 1

    def get_leaves_below(self):
        """Get all leaves below this leaf (inclusive)"""
        return [self]

    def update_bounds_below(self):
        """Update the bounds of the subtree rooted at this node"""
        return

    def pred(self, x):
        """Make a prediction for a single input sample."""
        return self.value


class Decision_Tree:
    """Class representing a decision tree"""
    def __init__(self, split_criterion="random",
                 max_depth=20, seed=0, min_pop=2):
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __str__(self):
        return self.root.__str__()

    def get_leaves(self):
        """Get all leaves in the decision tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update the bounds of the decision tree"""
        self.root.update_bounds_below()

    def pred(self, x):
        """Make a prediction for a single input sample."""
        return self.root.pred(x)

    def update_predict(self):
        """Update the prediction function for the decision tree."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            n = A.shape[0]
            y_pred = np.empty(n, dtype=object)
            mask_total = np.zeros(n, dtype=bool)
            for leaf in leaves:
                mask = leaf.indicator(A)
                y_pred[mask] = leaf.value
                mask_total = mask_total | mask
            if not np.all(mask_total):
                raise ValueError("Some rows in A are not covered by any leaf.")
            return y_pred.astype(int)
        self.predict = predict

    def fit(self, explanatory, target, verbose=0):
        """Fit the decision tree to the training data."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root = Node(is_root=True, depth=0)
        self.root.sub_population = np.ones_like(self.target, dtype='bool')
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(
                "  Training finished.\n"
                f"    - Depth                     : {self.depth()}\n"
                f"    - Number of nodes           : {self.count_nodes()}\n"
                f"    - Number of leaves          : "
                f"{self.count_nodes(only_leaves=True)}\n"
                f"    - Accuracy on training data : "
                f"{self.accuracy(self.explanatory, self.target)}"
            )

    def fit_node(self, node):
        """Fit a node in the decision tree."""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = (
            (self.explanatory[:, node.feature] > node.threshold)
            & node.sub_population
        )
        right_population = (
            (self.explanatory[:, node.feature] <= node.threshold)
            & node.sub_population
        )

        is_left_leaf = self.is_leaf(left_population,  node.depth + 1)
        is_right_leaf = self.is_leaf(right_population, node.depth + 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def is_leaf(self, sub_population, depth):
        """Check if a node is a leaf node."""
        if np.sum(sub_population) < self.min_pop:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return np.unique(self.target[sub_population]).size == 1

    def calculate_leaf_value(self, sub_population):
        """Calculate the value of a leaf node."""
        if np.sum(sub_population) == 0:
            vals, counts = np.unique(self.target, return_counts=True)
            return vals[np.argmax(counts)]
        vals, counts = np.unique(
            self.target[sub_population], return_counts=True
            )
        return vals[np.argmax(counts)]

    def get_leaf_child(self, node, sub_population):
        """Get the leaf child of a node."""
        value = self.calculate_leaf_value(sub_population)
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Get the node child of a node."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def random_split_criterion(self, node):
        """Randomly select a feature and a threshold for splitting."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            arr = self.explanatory[:, feature][node.sub_population]
            feature_min = np.min(arr)
            feature_max = np.max(arr)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_sub_population(self, node, left):
        """Get the sub-population for a node."""
        if left:
            return (
                self.explanatory[:, node.feature] > node.threshold
                ) & node.sub_population
        else:
            return (
                self.explanatory[:, node.feature] <= node.threshold
                ) & node.sub_population

    def depth(self):
        """Get the depth of the decision tree."""
        return self.root.max_depth_below() if hasattr(self, 'root') else 0

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes in the decision tree."""
        return self.root.count_nodes_below(
            only_leaves=only_leaves
            ) if hasattr(self, 'root') else 0

    def accuracy(self, test_explanatory, test_target):
        """Calculate the accuracy of the decision tree."""
        return (
            np.sum(
                np.equal(self.predict(test_explanatory), test_target)
            )
            / test_target.size
        )

    def possible_thresholds(self, node, feature):
        """Get the possible thresholds for a feature in a node."""
        values = np.unique(self.explanatory[:, feature][node.sub_population])
        if values.size < 2:
            return np.array([])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Return (best_threshold, min_weighted_gini) for a single feature."""
        X = self.explanatory[:, feature][node.sub_population]   # (n,)
        y = self.target[node.sub_population]                    # (n,)
        thresholds = self.possible_thresholds(node, feature)    # (t,)
        if thresholds.size == 0:
            return np.nan, np.inf

        classes = np.unique(y)                                  # (c,)

        X_col = X[:, None]                                      # (n,1)
        T_row = thresholds[None, :]                             # (1,t)
        left_mask = X_col > T_row                             # (n,t)
        right_mask = ~left_mask                                 # (n,t)

        C = (y[:, None] == classes[None, :]).astype(int)        # (n,c)

        left_counts = np.einsum('nt,nc->tc', left_mask,  C)
        right_counts = np.einsum('nt,nc->tc', right_mask, C)

        n_left = left_counts.sum(axis=1)
        n_right = right_counts.sum(axis=1)
        n_total = n_left + n_right

        left_props = np.divide(
            left_counts, n_left[:, None],
            out=np.zeros_like(left_counts, dtype=float),
            where=n_left[:, None] != 0
        )
        right_props = np.divide(
            right_counts, n_right[:, None],
            out=np.zeros_like(right_counts, dtype=float),
            where=n_right[:, None] != 0
        )

        gini_left = 1.0 - np.sum(left_props ** 2, axis=1)
        gini_right = 1.0 - np.sum(right_props ** 2, axis=1)

        weighted_gini = (n_left * gini_left + n_right * gini_right) / \
            np.maximum(n_total, 1)

        degenerate = (n_left == 0) | (n_right == 0)
        weighted_gini = np.where(degenerate, np.inf, weighted_gini)

        j = np.argmin(weighted_gini)
        return thresholds[j], weighted_gini[j]

    def Gini_split_criterion(self, node):
        """Compute the Gini impurity for a split on a single feature."""
        results = [
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ]
        thresholds = np.array([r[0] for r in results])
        ginis = np.array([r[1] for r in results])
        i = np.argmin(ginis)
        return i, thresholds[i]


class Random_Forest():
    """Class representing a random forest"""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.trees = []

    def predict(self, explanatory):
        """ Make a prediction for the given explanatory variables."""
        tree_predictions = []
        for tree in self.trees:
            tree_predictions.append(tree.predict(explanatory))
        return np.array([
            np.bincount(pred).argmax() for pred in zip(*tree_predictions)
        ])

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Fit the random forest model to the training data."""
        self.target = target
        self.explanatory = explanatory
        self.trees = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )
            T.fit(explanatory, target)
            self.trees.append(T)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(
                "  Training finished.\n"
                f"    - Mean depth                     : "
                f"{np.array(depths).mean()}\n"
                f"    - Mean number of nodes           : "
                f"{np.array(nodes).mean()}\n"
                f"    - Mean number of leaves          : "
                f"{np.array(leaves).mean()}\n"
                f"    - Mean accuracy on training data : "
                f"{np.array(accuracies).mean()}\n"
                f"    - Accuracy of the forest on td   : "
                f"{self.accuracy(self.explanatory, self.target)}"
            )

    def accuracy(self, test_explanatory, test_target):
        """Compute the accuracy of the model on the test data."""
        return (
            np.sum(np.equal(self.predict(test_explanatory), test_target)) /
            test_target.size
        )


def left_child_add_prefix(text):
    """Add a prefix to the left child representation"""
    lines = text.split("\n")
    new_text = "    +---> " + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |  " + x) + "\n"
    return new_text


def right_child_add_prefix(text):
    """Add a prefix to the right child representation"""
    lines = text.split("\n")
    new_text = "    +---> " + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("       " + x) + "\n"
    return new_text


if __name__ == "__main__":
    T = Decision_Tree(split_criterion="random", max_depth=20, seed=0)
    T.fit(explanatory, target)
