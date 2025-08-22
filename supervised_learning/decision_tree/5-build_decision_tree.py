#!/usr/bin/env python3
"""Module for building a decision tree"""

from __future__ import annotations
import numpy as np


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
            if not hasattr(self, 'lower') or len(getattr(self, 'lower', {})) == 0:
                return np.ones(x.shape[0], dtype=bool)
            conds = [x[:, k] > self.lower[k] for k in self.lower]
            return np.all(np.stack(conds, axis=0), axis=0) if conds else np.ones(x.shape[0], bool)

        def is_small_enough(x):
            if not hasattr(self, 'upper') or len(getattr(self, 'upper', {})) == 0:
                return np.ones(x.shape[0], dtype=bool)
            conds = [x[:, k] <= self.upper[k] for k in self.upper]
            return np.all(np.stack(conds, axis=0), axis=0) if conds else np.ones(x.shape[0], bool)

        self.indicator = lambda x: np.all(
            np.stack([is_large_enough(x), is_small_enough(x)], axis=0), axis=0
        )


class Leaf(Node):
    """Class representing a leaf node in the decision tree"""
    def __init__(self, value, depth=0, is_root=False):
        self.value = value
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = True

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


class Decision_Tree:
    """Class representing a decision tree"""
    def __init__(self, root):
        self.root = root

    def __str__(self):
        return self.root.__str__()

    def get_leaves(self):
        """Get all leaves in the decision tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update the bounds of the decision tree"""
        self.root.update_bounds_below()


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
