class Trie:
    def __init__(self):
        self.trie = {}
    def insert(self, itemset):
        """
        Insert a sorted itemset into the Trie.
        """
        current_node = self.trie
        for item in itemset:  # Assume itemset is pre-sorted
            if item not in current_node:
                current_node[item] = {}
            current_node = current_node[item]
        # Mark the end of this itemset in the Trie
        current_node['*'] = current_node.get('*', 0) + 1
    def get_support(self, itemset):
        """
        Retrieve the support count for a sorted itemset from the Trie.
        """
        current_node = self.trie
        for item in itemset:  # Assume itemset is pre-sorted
            if item not in current_node:
                return 0
            current_node = current_node[item]
        return current_node.get('*', 0)
    def generate_candidates(self, current_level_itemsets):
        """
        Generate candidate itemsets of size k+1 from frequent k-itemsets
        using the Trie for efficient prefix matching.
        """
        candidates = set()
        frequent_items = list(current_level_itemsets)
        len_frequent_items = len(frequent_items)

        for i in range(len_frequent_items):
            for j in range(i + 1, len_frequent_items):
                l1 = sorted(frequent_items[i])  # Assume frequent_items[i] is sorted
                l2 = sorted(frequent_items[j])  # Assume frequent_items[j] is sorted

                # Check if first k-1 items are the same
                if l1[:-1] == l2[:-1]:
                    # Combine the two sets into a new candidate
                    candidate = frozenset(frequent_items[i] | frequent_items[j])

                    # Validate that all subsets of size k are frequent
                    if all(frozenset(candidate - {item}) in current_level_itemsets for item in candidate):
                        candidates.add(candidate)

        return candidates
    def compress(self):
        """
        Compress the Trie by collapsing single-child nodes.
        """
        def compress_node(node):
            keys = list(node.keys())
            if len(keys) == 1 and keys[0] != '*':
                # Collapse the node with its child
                child = node[keys[0]]
                if isinstance(child, dict):
                    compressed_child = compress_node(child)
                    return {keys[0]: compressed_child}
            else:
                for key in keys:
                    if isinstance(node[key], dict):
                        node[key] = compress_node(node[key])
            return node

        self.trie = compress_node(self.trie)