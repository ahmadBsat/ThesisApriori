class Trie:
    def __init__(self):
        self.trie = {}

    def insert(self, itemset):
        """Insert an itemset into the trie."""
        current_node = self.trie
        for item in sorted(itemset):
            if item not in current_node:
                current_node[item] = {}
            current_node = current_node[item]
        # Mark the end of this itemset in the trie
        current_node['*'] = current_node.get('*', 0) + 1

    def get_support(self, itemset):
        """Get the support count for an itemset from the trie."""
        current_node = self.trie
        for item in sorted(itemset):
            if item not in current_node:
                return 0
            current_node = current_node[item]
        return current_node.get('*', 0)

    def generate_candidates(self, current_level_itemsets):
        """Generate candidate itemsets of size k+1 from frequent k-itemsets."""
        candidates = set()
        frequent_items = list(current_level_itemsets)
        len_frequent_items = len(frequent_items)

        for i in range(len_frequent_items):
            for j in range(i + 1, len_frequent_items):
                # Convert frozenset to sorted lists for comparison
                l1 = sorted(frequent_items[i])
                l2 = sorted(frequent_items[j])

                # Check if first k-1 items are the same
                if l1[:-1] == l2[:-1]:
                    # Combine the two sets into a new candidate
                    candidate = frozenset(frequent_items[i] | frequent_items[j])
                    candidates.add(candidate)

        return candidates
