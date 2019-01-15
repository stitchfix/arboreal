from core.node import Node


def _str__helper(node, depth=0):
    prefix = "\n" + "\t" * depth
    body = prefix + str(node)

    # base case, there are no children:
    if not node.children:
        return body

    # recursive case, there are children:
    elif node.children:
        for child in node.children:
            body += _str__helper(node.children[child], depth=depth + 1)
        return body


class Tree:
    def __init__(self, root=None, name=None):
        if not root:
            self._root = Node()
        else:
            self._root = root
        self._name = name

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, val):
        self._root = val

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    def __str__(self):
        ret = f"<Tree '{self.name}'>"
        ret += _str__helper(node=self.root)
        return ret
