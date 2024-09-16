import time
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain, combinations
from TrieClass import Trie, TrieNode

# Function to load transactions from file
def get_transactions_from_file(file_path):
    print("Reading the file...")
    df = pd.read_csv(file_path)
    grouped = df.groupby('Member_number')['itemDescription'].apply(list)
    transactions = grouped.tolist()
    print("Number of transactions found:", len(transactions))
    return transactions

# Function to prune candidates based on minimum support
def prune_candidates(support_count, min_support):
    print(f"Pruning candidates with support < {min_support}...")
    return {k: v for k, v in support_count.items() if v >= min_support}

# Apriori algorithm using Trie
def apriori_with_trie(min_support, file_path, max_itemset_size=3):
    transactions = get_transactions_from_file(file_path)
    transaction_list = list(map(set, transactions))

    trie = Trie()
    frequent_itemsets = {}
    loop_count = 1
    loop_times = []

    # Insert all itemsets into the Trie
    for transaction in transaction_list:
        for length in range(1, len(transaction) + 1):
            for itemset in combinations(transaction, length):
                trie.insert(sorted(itemset))

    # Start timing
    start_time = time.time()

    while loop_count <= max_itemset_size:
        candidates = []
        trie.generate_candidates([], trie.root, loop_count, candidates)

        support_count = {tuple(candidate): trie.get_support(candidate) for candidate in candidates}
        frequent_k_itemsets = prune_candidates(support_count, min_support)

        if not frequent_k_itemsets:
            print(f"No more frequent itemsets found in loop {loop_count}. Ending...")
            break

        frequent_itemsets.update(frequent_k_itemsets)
        loop_count += 1

        # Track time for this loop
        loop_end_time = time.time()
        loop_time = loop_end_time - start_time
        loop_times.append(loop_time)

    # End timing
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    return frequent_itemsets, loop_times

# Function to generate association rules from frequent itemsets
def generate_rules(frequent_itemsets, transaction_list, min_confidence=0.6):
    rules = []

    for itemset in frequent_itemsets.keys():
        length = len(itemset)
        if length > 1:  # Rules can only be generated for itemsets of size 2 or more
            subsets = chain(*[combinations(itemset, i) for i in range(1, length)])
            for antecedent in subsets:
                antecedent = frozenset(antecedent)
                consequent = frozenset(itemset) - antecedent

                # Check if antecedent and consequent are frequent
                if antecedent in frequent_itemsets and consequent in frequent_itemsets:
                    confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                    support = frequent_itemsets[itemset] / len(transaction_list)
                    lift = confidence / (frequent_itemsets[consequent] / len(transaction_list))

                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, support, confidence, lift))
                else:
                    # Skip rule generation if subsets are not frequent
                    print(
                        f"Skipping rule for itemset {itemset} as subset {antecedent} or {consequent} is not frequent.")

    return rules


# Function to visualize the frequent itemsets and loop times
def visualize_frequent_itemsets(frequent_itemsets, loop_times):
    # Convert the frequent itemsets to a pandas DataFrame for easy visualization
    itemsets = [' & '.join(itemset) for itemset in frequent_itemsets.keys()]
    support_counts = list(frequent_itemsets.values())
    df = pd.DataFrame({'Itemset': itemsets, 'Support Count': support_counts})
    df = df.sort_values(by='Support Count', ascending=False).head(10)  # Show top 10

    plt.figure(figsize=(14, 6))

    # Table for frequent itemsets
    plt.subplot(1, 2, 1)
    plt.axis('off')
    table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title('Top 10 Frequent Itemsets')

    # Bar chart for loop times
    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(loop_times) + 1), loop_times, color='lightcoral')
    plt.xlabel('Loop (Itemset Size)')
    plt.ylabel('Time (seconds)')
    plt.title('Time Taken per Loop')

    plt.tight_layout()
    plt.show()

# Function to save frequent itemsets to a CSV file
def save_frequent_itemsets_to_csv(frequent_itemsets, file_name='frequent_itemsets.csv'):
    itemsets = [' & '.join(itemset) for itemset in frequent_itemsets.keys()]
    support_counts = list(frequent_itemsets.values())
    df = pd.DataFrame({'Itemset': itemsets, 'Support Count': support_counts})
    df = df.sort_values(by='Support Count', ascending=False)  # Sort by support count
    df.to_csv(file_name, index=False)
    print(f"Frequent itemsets saved to {file_name}")

# Function to save association rules to a CSV file
def save_rules_to_csv(rules, file_name='association_rules.csv'):
    df = pd.DataFrame(rules, columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
    df['Antecedent'] = df['Antecedent'].apply(lambda x: ', '.join(list(x)))
    df['Consequent'] = df['Consequent'].apply(lambda x: ', '.join(list(x)))
    df.to_csv(file_name, index=False)
    print(f"Association rules saved to {file_name}")

# Set a minimum support and confidence threshold
min_support = 2
min_confidence = 0.7

# Provide the correct file path
file_path = 'Groceries_dataset.csv'

# Run the Apriori algorithm with Trie on the file data
print("Running the Apriori algorithm with Trie...")
frequent_itemsets, loop_times = apriori_with_trie(min_support, file_path)

# Save the frequent itemsets to a CSV file
save_frequent_itemsets_to_csv(frequent_itemsets)

# Ensure transaction_list is passed correctly
transactions = get_transactions_from_file(file_path)
transaction_list = list(map(set, transactions))

# Generate and save the association rules
rules = generate_rules(frequent_itemsets, transaction_list, min_confidence)
save_rules_to_csv(rules)

# Visualize the results
visualize_frequent_itemsets(frequent_itemsets, loop_times)

