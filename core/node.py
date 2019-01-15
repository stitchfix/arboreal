class Node:
    def __init__(self, name=None, data=None):
        self._children = dict()
        self._parent = None
        self._data = data
        self._name = name

        # if no data is passed in, default to a dict
        if not data:
            self._data = dict()

    @property
    def children(self):
        return self._children

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, val):
        self._parent = val

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def is_leaf(self):
        return not bool(len(self.children) >= 1)

    def add_child(self, child):
        child.parent = self
        if child.name:
            self._children[child.name] = child
        else:
            default_name = repr(child)
            self._children[default_name] = child

    def remove_child(self, name):
        self._children[name].parent = None
        del self._children[name]

    def __str__(self):
        return f"<Node {self.name}>"
