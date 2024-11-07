import time
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool
from TrieClass import Trie


# Function to load transactions from file
# This function reads a file with transactions (purchases), pulls out the list of items bought in each transaction,
# and organizes them into sets for easy processing later.
def get_transactions_from_file(file_path):
    df = pd.read_csv(file_path)
    item_columns = [col for col in df.columns if col.startswith('Item ')]
    transactions = []
    for idx, row in df.iterrows():
        items = row[item_columns].dropna().astype(str).tolist()
        items = [item.strip() for item in items if item.strip()]
        transactions.append(set(items))
    return transactions


# Helper function to count how often items appear in transactions
# This function counts how many times groups of items (candidates) are bought together in different transactions.
def count_support_parallel(transactions, candidates):
    count_dict = {candidate: 0 for candidate in candidates}
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                count_dict[candidate] += 1
    return count_dict


# Function to count how often items appear using multiple processors
# This function splits up the transactions and has multiple parts of the computer work on counting them at the same time,
# making the process faster.
def parallel_support_counting(transactions, candidates, min_support, num_transactions):
    # Divide transactions into chunks for parallel processing
    num_processes = 4  # Adjust based on your CPU cores
    chunk_size = len(transactions) // num_processes
    transaction_chunks = [transactions[i:i + chunk_size] for i in range(0, len(transactions), chunk_size)]

    # Use a multiprocessing pool
    pool = Pool(processes=num_processes)
    results = pool.starmap(count_support_parallel, [(chunk, candidates) for chunk in transaction_chunks])

    # Combine results from all processes
    total_counts = {}
    for result in results:
        for candidate, count in result.items():
            total_counts[candidate] = total_counts.get(candidate, 0) + count

    # Calculate support and prune candidates below min_support
    itemset_counts = {}
    for candidate, count in total_counts.items():
        support = count / num_transactions
        if support >= min_support:
            itemset_counts[candidate] = support

    pool.close()
    pool.join()

    return itemset_counts


# Function to filter out the items that aren’t popular enough
# This function checks which groups of items are bought often enough to be considered "frequent"
# and removes the ones that aren't bought frequently.
def prune_candidates(trie, candidates, min_support, num_transactions):
    frequent_itemsets = {}
    for candidate in candidates:
        support_count = trie.get_support(candidate)
        if support_count >= min_support * num_transactions:
            support = support_count / num_transactions
            frequent_itemsets[candidate] = support
    return frequent_itemsets


# Optimized Apriori algorithm using Trie with parallel support counting
# This function finds out which items or groups of items are bought together frequently. It starts with individual items,
# then combines them to check for bigger groups that are also frequent.
def apriori_with_trie(transactions, min_support):
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions
    trie = Trie()
    frequent_itemsets = {}
    loop_count = 1
    current_level_itemsets = set()

    # Initialize with frequent 1-itemsets
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    for item, count in item_counts.items():
        if count >= min_support_count:
            candidate = frozenset([item])
            trie.insert(candidate)
            support = count / num_transactions
            frequent_itemsets[candidate] = support
            current_level_itemsets.add(candidate)

    # Start timing
    start_time = time.time()

    # Iteratively generate and prune itemsets of increasing size
    while current_level_itemsets:
        print(f"Iteration {loop_count}: Generating candidates of size {loop_count + 1}")
        # Generate candidates of size k+1 from frequent k-itemsets
        candidates = trie.generate_candidates(current_level_itemsets)
        if not candidates:
            print("No new candidates generated.")
            break

        # Count supports of candidates using parallel processing
        frequent_itemsets_k = parallel_support_counting(transactions, candidates, min_support, num_transactions)

        if not frequent_itemsets_k:
            print(f"No more frequent itemsets found in iteration {loop_count}. Ending...")
            break

        # Update frequent itemsets
        frequent_itemsets.update(frequent_itemsets_k)
        current_level_itemsets = set(frequent_itemsets_k.keys())
        loop_count += 1

    # End timing
    elapsed_time = time.time() - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Number of frequent itemsets found: {len(frequent_itemsets)}")

    return frequent_itemsets


# Function to create rules from frequent itemsets
# This function creates "if-then" style rules from the frequent items,
# like "if someone buys X, they often also buy Y" and calculates how often the rule is true.
def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets.keys():
        if len(itemset) >= 2:
            itemset_support = frequent_itemsets[itemset]
            # Generate all non-empty proper subsets of the itemset
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if antecedent in frequent_itemsets and consequent in frequent_itemsets:
                        antecedent_support = frequent_itemsets[antecedent]
                        confidence = itemset_support / antecedent_support
                        if confidence >= min_confidence:
                            consequent_support = frequent_itemsets[consequent]
                            lift = confidence / consequent_support if consequent_support > 0 else 0
                            rules.append({
                                'Antecedent': ', '.join(antecedent),
                                'Consequent': ', '.join(consequent),
                                'Support': itemset_support,
                                'Confidence': confidence,
                                'Lift': lift
                            })
    print(f"Generated {len(rules)} association rules")
    return rules


# Function to visualize the frequent itemsets and loop times
# This function shows the top 10 most frequently bought groups of items as a bar chart to make it easier to understand.
def visualize_frequent_itemsets(frequent_itemsets):
    itemsets = [' & '.join(itemset) for itemset in frequent_itemsets.keys()]
    supports = list(frequent_itemsets.values())
    df = pd.DataFrame({'Itemset': itemsets, 'Support': supports})
    df = df.sort_values(by='Support', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(df['Itemset'], df['Support'], color='skyblue')
    plt.xlabel('Support')
    plt.ylabel('Itemset')
    plt.title('Top 10 Frequent Itemsets')
    plt.gca().invert_yaxis()
    plt.show()


# Function to save frequent itemsets to a CSV file
# This function saves the frequent items (or groups of items) to a file so you can look at them later.
def save_frequent_itemsets_to_csv(frequent_itemsets, file_name='frequent_itemsets_trie.csv'):
    itemsets = [' & '.join(itemset) for itemset in frequent_itemsets.keys()]
    supports = list(frequent_itemsets.values())
    df = pd.DataFrame({'Itemset': itemsets, 'Support': supports})
    df.to_csv(file_name, index=False)
    print(f"Frequent itemsets saved to {file_name}")


# Function to save association rules to a CSV file
# This function saves the "if-then" style rules to a file so you can look at them later or share them.
def save_rules_to_csv(rules, file_name='association_rules_trie.csv'):
    df = pd.DataFrame(rules)
    df.to_csv(file_name, index=False)
    print(f"Association rules saved to {file_name}")


if __name__ == "__main__":
    # Set a minimum support ratio and confidence threshold
    # These are like settings to tell the algorithm how often a group of items should appear to be considered frequent,
    # and how strong the "if-then" rules should be.
    min_support_ratio = 0.01  # 1%
    min_confidence = 0.2  # Lowered from 0.6 to generate more rules

    # Provide the correct file path
    file_path = 'groceries-groceries.csv'  # Update with your file path

    # Load transactions
    # This step loads the data from the file.
    transactions = get_transactions_from_file(file_path)

    # Run the optimized Apriori algorithm with Trie on the file data
    # This step runs the process of finding frequent items.
    frequent_itemsets = apriori_with_trie(transactions, min_support_ratio)

    # Save the frequent itemsets to a CSV file
    # This saves the frequent items found to a file.
    save_frequent_itemsets_to_csv(frequent_itemsets)

    # Generate and save the association rules
    # This creates and saves the "if-then" rules to a file.
    rules = generate_rules(frequent_itemsets, min_confidence)
    save_rules_to_csv(rules)

    # Visualize the results
    # This step shows the most frequent items as a bar chart.
    visualize_frequent_itemsets(frequent_itemsets)

