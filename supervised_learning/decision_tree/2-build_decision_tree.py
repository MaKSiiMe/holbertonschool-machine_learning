class Leaf:
    def __init__(self, value, depth=0, is_root=False):
        self.value = value
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = True

    def __str__(self):
        return (f"-> leaf [value={self.value}]")


class Node:
    def __init__(self, feature, threshold, left_child=None, right_child=None, depth=0, is_root=False):
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
            parts.append(left_child_add_prefix(str(self.left_child)).rstrip("\n"))
        if self.right_child:
            parts.append(right_child_add_prefix(str(self.right_child)).rstrip("\n"))
        return "\n".join(parts)


class Decision_Tree:
    def __init__(self, root):
        self.root = root

    def __str__(self):
        return self.root.__str__()


def left_child_add_prefix(text):
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |  " + x) + "\n"
    return new_text


def right_child_add_prefix(text):
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("       " + x) + "\n"
    return new_text
