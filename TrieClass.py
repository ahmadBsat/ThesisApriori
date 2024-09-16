class TrieNode:
    def __init__(self):
        self.children = {}
        self.support = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, itemset):
        node = self.root
        for item in itemset:
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]
        node.support += 1

    def get_support(self, itemset):
        node = self.root
        for item in itemset:
            if item in node.children:
                node = node.children[item]
            else:
                return 0
        return node.support

    def generate_candidates(self, itemset_prefix, node, length, result):
        if length == 0:
            result.append(itemset_prefix)
            return
        for item, child_node in node.children.items():
            self.generate_candidates(itemset_prefix + [item], child_node, length - 1, result)
