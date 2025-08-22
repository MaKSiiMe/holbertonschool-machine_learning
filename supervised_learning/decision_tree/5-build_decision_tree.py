#!/usr/bin/env python3
"""Module for building a decision tree"""

from __future__ import annotations
import numpy as np


class Leaf:
	"""Leaf node (stocke une valeur et recevra bornes + indicateur)."""
	def __init__(self, value, depth=0, is_root=False):
		self.value = value
		self.depth = depth
		self.is_root = is_root
		self.is_leaf = True

	def __str__(self):
		return f"-> leaf [value={self.value}]"

	def get_leaves_below(self):
		return [self]


class Node:
	def __init__(
		self, feature, threshold, *, left_child=None,
		right_child=None, depth=0, is_root=False
	):
		self.feature = feature
		self.threshold = threshold
		self.left_child = left_child
		self.right_child = right_child
		self.depth = depth
		self.is_root = is_root
		self.is_leaf = False

	def __str__(self):
		root_line = f"[feature={self.feature}, threshold={self.threshold}]"
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

	def get_leaves_below(self):
		leaves = []
		if self.left_child:
			leaves.extend(self.left_child.get_leaves_below())
		if self.right_child:
			leaves.extend(self.right_child.get_leaves_below())
		return leaves

	def update_bounds_below(self):
		if self.is_root:
			self.lower = {}
			self.upper = {}
		else:
			if not hasattr(self, 'lower'):
				self.lower = {}
			if not hasattr(self, 'upper'):
				self.upper = {}

		if self.left_child is not None:
			self.left_child.lower = dict(self.lower)
			self.left_child.upper = dict(self.upper)
			self.left_child.lower[self.feature] = self.threshold
		if self.right_child is not None:
			self.right_child.lower = dict(self.lower)
			self.right_child.upper = dict(self.upper)
			self.right_child.upper[self.feature] = self.threshold
			if self.is_root and self.feature not in self.right_child.lower:
				self.right_child.lower[self.feature] = -np.inf

		if self.left_child is not None and not self.left_child.is_leaf:
			self.left_child.update_bounds_below()
		if self.right_child is not None and not self.right_child.is_leaf:
			self.right_child.update_bounds_below()

	def update_indicator(self):
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


Leaf.update_indicator = Node.update_indicator


class Decision_Tree:
	def __init__(self, root):
		self.root = root

	def __str__(self):
		return self.root.__str__()

	def get_leaves(self):
		return self.root.get_leaves_below()

	def update_bounds(self):
		self.root.update_bounds_below()
		for leaf in self.get_leaves():
			if not hasattr(leaf, 'lower'):
				leaf.lower = {}
			if not hasattr(leaf, 'upper'):
				leaf.upper = {}
			for f in list(leaf.lower.keys()):
				if f not in leaf.upper:
					leaf.upper[f] = np.inf


def left_child_add_prefix(text):
	lines = text.split("\n")
	new_text = "    +--" + lines[0] + "\n"
	for x in lines[1:]:
		new_text += "    |  " + x + "\n"
	return new_text


def right_child_add_prefix(text):
	lines = text.split("\n")
	new_text = "    \\--" + lines[0] + "\n"
	for x in lines[1:]:
		new_text += "       " + x + "\n"
	return new_text
