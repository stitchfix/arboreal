from nose_focus import focus  # enable with @focus
import unittest

from core.node import Node


class TestNode(unittest.TestCase):
    @focus
    def test_node_constructor(self):
        n = Node()
        self.assertEqual(n.children, dict())
        self.assertIsNone(n.parent)
        self.assertTrue(isinstance(n.data, dict))

    def test_node_add_then_get_one_child(self):
        n = Node()
        left = Node("left")
        n.add_child(left)
        self.assertEqual(n.children["left"], left)

    def test_node_add_child_sets_child_parent(self):
        n = Node(data=1)
        left = Node(name="left", data=2)
        n.add_child(left)
        self.assertIsNotNone(n.children["left"].parent)
        self.assertEqual(n.children["left"].parent.data, 1)

    def test_node_add_then_get_children(self):
        n = Node()
        left = Node()
        right = Node()
        n.add_child(left)
        n.add_child(right)
        self.assertEqual(set(n.children.values()), set([left, right]))

    def test_node_add_then_delete_one_child(self):
        n = Node()
        left = Node("left")
        n.add_child(left)
        n.remove_child("left")
        self.assertEqual(n.children, dict())

    def test_node_add_multiple_children_then_delete_one(self):
        n = Node()
        left = Node("left")
        right = Node("right")
        n.add_child(left)
        n.add_child(right)
        n.remove_child("left")
        self.assertEqual(n.children, dict(right=right))

    def test_node_add_parent(self):
        n = Node()
        parent = Node()
        n.parent = parent
        self.assertEqual(n.parent, parent)

    def test_node_modify_parent(self):
        n = Node()
        parent = Node(data=1)
        n.parent = parent
        self.assertEqual(n.parent, parent)
        new_parent = Node(data=2)
        n.parent = new_parent
        self.assertEqual(n.parent, new_parent)

    def test_node_delete_parent(self):
        n = Node()
        parent = Node()
        n.parent = parent
        self.assertEqual(n.parent, parent)
        n.parent = None
        self.assertEqual(n.parent, None)
