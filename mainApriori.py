import os
import pandas as pd
from itertools import combinations
from collections import Counter
import time
import logging
import psutil
import tracemalloc

# Configure logging format (prints to console)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_all_itemsets(items, length):
    """
    Generate all possible itemsets of a given length (brute force, no pruning).
    """
    return [frozenset(itemset) for itemset in combinations(items, length)]


def count_support(transactions, itemsets):
    """
    Efficient counting of support for itemsets using Counter.
    """
    itemset_counts = Counter()
    for transaction in transactions:
        for itemset in itemsets:
            if itemset.issubset(transaction):
                itemset_counts[itemset] += 1
    return itemset_counts


def apriori_brute_force(transactions, min_support):
    """
    Basic version of the Apriori algorithm (unbounded itemset size).
    Continues until no more frequent itemsets can be found.
    """
    items = set(item for transaction in transactions for item in transaction)
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions

    frequent_itemsets = {}
    iteration = 1

    while True:
        logging.info(f"Iteration {iteration}: Generating candidates of size {iteration + 1}")
        # Generate all possible itemsets of size (iteration+1)
        itemsets = get_all_itemsets(items, iteration + 1)
        if not itemsets:
            # If no candidates can be generated, break
            logging.info("No new candidates generated.")
            break

        # Count the support of all itemsets
        itemset_counts = count_support(transactions, itemsets)

        current_frequent_itemsets = {
            itemset: count / num_transactions
            for itemset, count in itemset_counts.items()
            if count >= min_support_count
        }

        if not current_frequent_itemsets:
            # If none of the candidates are frequent, stop
            logging.info("No new candidates generated.")
            break

        # Update the global frequent itemsets dictionary
        frequent_itemsets.update(current_frequent_itemsets)
        iteration += 1

    return frequent_itemsets


def generate_rules_brute_force(frequent_itemsets, min_confidence):
    """
    Generates association rules without optimization:
    brute force over all combinations of frequent itemsets.
    """
    start_time = time.time()
    rules = []

    for itemset in frequent_itemsets:
        if len(itemset) >= 2:
            # Generate all non-empty subsets of the itemset
            subsets = [
                frozenset(x)
                for r in range(1, len(itemset))
                for x in combinations(itemset, r)
            ]
            for antecedent in subsets:
                consequent = itemset - antecedent
                if consequent:
                    antecedent_support = frequent_itemsets.get(antecedent, 0)
                    if antecedent_support > 0:
                        confidence = frequent_itemsets[itemset] / antecedent_support
                        if confidence >= min_confidence:
                            consequent_support = frequent_itemsets.get(consequent, 0)
                            lift = (confidence / consequent_support) if consequent_support > 0 else 0
                            support = frequent_itemsets[itemset]
                            rules.append({
                                'Antecedent': ', '.join(antecedent),
                                'Consequent': ', '.join(consequent),
                                'Support': support,
                                'Confidence': confidence,
                                'Lift': lift
                            })

    _ = time.time() - start_time
    return rules


def get_transactions_from_file(file_path):
    """
    Reads the dataset from a text file where each line represents a transaction.
    Handles missing values and converts each transaction into a list of items.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convert each line into a transaction (list of items)
    transactions = [line.strip().split() for line in lines if line.strip()]

    return transactions


if __name__ == "__main__":
    # Settings you can adjust
    folder_path = 'SingleDataset'   # Folder containing all your .txt files
    min_support = 0.01              # Minimum support as a fraction (e.g., 1%)
    min_confidence = 0.1            # Minimum confidence threshold

    # Collect all .txt files from the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)

        # Start memory and time tracking
        tracemalloc.start()
        start_time = time.time()

        logging.info(f"\nProcessing dataset: {file_path}")
        transactions = get_transactions_from_file(file_path)

        # Run the brute-force Apriori algorithm (unbounded itemset size)
        frequent_itemsets = apriori_brute_force(transactions, min_support)

        # Generate association rules
        _ = generate_rules_brute_force(frequent_itemsets, min_confidence)

        # End memory and time tracking
        total_time = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Convert bytes to MB
        memory_info = psutil.virtual_memory()
        used_memory_mb = memory_info.used / (1024 * 1024)
        peak_memory_mb = peak_mem / (1024 * 1024)

        logging.info(f"Total time taken: {total_time:.2f} seconds")
        logging.info(f"Memory used (psutil): {used_memory_mb:.2f} MB")
        logging.info(f"Peak memory (tracemalloc): {peak_memory_mb:.2f} MB")
        logging.info(f"Number of frequent itemsets found: {len(frequent_itemsets)}")
