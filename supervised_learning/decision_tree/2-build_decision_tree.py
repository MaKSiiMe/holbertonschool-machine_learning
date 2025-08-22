#!/usr/bin/env python3
"""Module for building a decision tree"""
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


class Leaf(Node):
    """Class representing a leaf node in the decision tree"""
    def __init__(self, value, depth=0, is_root=False):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth
        self.is_root = is_root

    def max_depth_below(self):
        """Calculate the maximum depth of the subtree rooted at this leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes in the subtree rooted at this leaf"""
        return 1

    def __str__(self):
        return f"leaf [value={self.value}]"


class Decision_Tree:
    """Class representing a decision tree"""
    def __init__(self, root):
        self.root = root

    def __str__(self):
        return self.root.__str__()


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
