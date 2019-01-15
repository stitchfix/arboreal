from nose_focus import focus  # enable with @focus
import unittest

from core.node import Node
from core.tree import Tree


class TestTree(unittest.TestCase):
    def test_tree_constructor(self):
        t = Tree(name="spruce")
        self.assertEqual(t.name, "spruce")
        self.assertIsNotNone(t.root)

    def test_tree_print_just_root(self):
        t = Tree()
        expected = "<Tree 'None'>\n<Node None>"
        self.assertEqual(str(t), expected)

    def test_tree_print_with_one_child(self):
        t = Tree()
        left = Node(name="left")
        t.root.add_child(left)
        expected = "<Tree 'None'>\n<Node None>\n\t<Node left>"
        self.assertEqual(str(t), expected)

    def test_tree_print_with_two_children(self):
        root = Node(name="first_level")
        t = Tree(root=root, name="root")
        left = Node(name="second_level_one")
        right = Node(name="second_level_two")
        t.root.add_child(left)
        t.root.add_child(right)
        expected = (
            "<Tree 'root'>\n<Node first_level>\n\t<Node second_level_one>\n\t"
        )
        expected += "<Node second_level_two>"  # ...
        self.assertEqual(str(t), expected)

    def test_tree_print_with_several_children(self):
        root = Node(name="first_level")
        t = Tree(root=root, name="root")
        left = Node(name="second_level_one")
        right = Node(name="second_level_two")
        t.root.add_child(left)
        t.root.add_child(right)
        left.add_child(Node("third_level_one"))
        left.add_child(Node("third_level_two"))
        left.add_child(Node("third_level_three"))
        expected = "<Tree 'root'>\n<Node first_level>\n\t<Node second_level_one>\n\t\t"
        expected += "<Node third_level_one>\n\t\t<Node third_level_two>\n\t\t<Node third_level_three>"  # ...
        expected += "\n\t<Node second_level_two>"  # ...
        self.assertEqual(str(t), expected)
