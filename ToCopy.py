import matplotlib.pyplot as plt
import networkx as nx

# Define the Trie structure as a dictionary
trie_structure = {
    "Root": {
        "Butter": {
            "*": 1,
            "Bread": {"*": 1},
            "Milk": {"*": 1},
            "Jam": {"*": 1},
            "Eggs": {"*": 1}
        },
        "Bread": {
            "*": 1,
            "Milk": {"*": 1},
            "Eggs": {"*": 1}
        },
        "Milk": {
            "*": 1,
            "Jam": {"*": 1}
        },
        "Eggs": {
            "*": 1
        },
        "Jam": {
            "*": 1
        }
    }
}



# Function to draw a readable Trie structure
def draw_readable_trie(trie, root_label="Root", file_path="Trie_Structure.png"):
    G = nx.DiGraph()

    def add_edges(node, parent_label):
        for key, value in node.items():
            if key == "*":
                continue  # Skip support count
            current_label = f"{key}"
            G.add_edge(parent_label, current_label)
            if isinstance(value, dict):
                add_edges(value, current_label)

    add_edges(trie, root_label)

    pos = nx.shell_layout(G)  # Use shell layout for clarity
    plt.figure(figsize=(16, 12))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="black", font_size=12, font_weight="bold",
            node_size=2000)
    plt.title("Readable Trie Structure for Frequent Itemsets", fontsize=16)
    plt.savefig(file_path)
    plt.show()


# Draw and save the Trie structure
draw_readable_trie(trie_structure)
