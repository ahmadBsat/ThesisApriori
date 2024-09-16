import time
import pandas as pd
import numpy as np
from itertools import chain, combinations
from multiprocessing import Pool
from collections import defaultdict, Counter
import tracemalloc

def get_transactions_from_file(file_path):
    df = pd.read_csv(file_path)
    grouped = df.groupby('Member_number')['itemDescription'].apply(list)
    transactions = grouped.tolist()
    return list(map(set, transactions))  # Convert to set for faster lookups

def get_candidates(transaction_list, length):
    # Optimized candidate generation
    candidates = set()
    for transaction in transaction_list:
        for combo in combinations(transaction, length):
            candidates.add(combo)
    return list(candidates)

def count_support_single(args):
    candidate, transaction_list = args
    return sum(1 for transaction in transaction_list if candidate.issubset(transaction))

def count_support(candidates, transaction_list):
    # Prepare arguments for multiprocessing
    args = [(set(candidate), transaction_list) for candidate in candidates]

    # Parallelized support counting
    with Pool() as pool:
        supports = pool.map(count_support_single, args)
    return dict(zip(candidates, supports))

def prune_candidates(support_count, min_support):
    return {k: v for k, v in support_count.items() if v >= min_support}

def apriori_algorithm(min_support, file_path, max_itemset_size=3):
    transactions = get_transactions_from_file(file_path)
    frequent_itemsets = {}
    k = 1

    tracemalloc.start()
    start_time = time.time()

    while k <= max_itemset_size:
        candidates = get_candidates(transactions, k)
        support_count = count_support(candidates, transactions)
        frequent_k_itemsets = prune_candidates(support_count, min_support)

        if not frequent_k_itemsets:
            break

        frequent_itemsets.update(frequent_k_itemsets)
        k += 1

    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Memory usage: {peak / 10**6:.2f} MB")

    return frequent_itemsets

def generate_rules(frequent_itemsets, transaction_list, min_confidence=0.6):
    rules = []
    for itemset in frequent_itemsets.keys():
        length = len(itemset)
        if length > 1:
            subsets = chain(*[combinations(itemset, i) for i in range(1, length)])
            for antecedent in subsets:
                antecedent = frozenset(antecedent)
                consequent = frozenset(itemset) - antecedent
                if antecedent in frequent_itemsets and consequent in frequent_itemsets:
                    confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                    support = frequent_itemsets[itemset] / len(transaction_list)
                    lift = confidence / (frequent_itemsets[consequent] / len(transaction_list))
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, support, confidence, lift))
    return rules

def visualize_frequent_itemsets(frequent_itemsets):
    df = pd.DataFrame({
        'Itemset': [' & '.join(itemset) for itemset in frequent_itemsets.keys()],
        'Support Count': list(frequent_itemsets.values())
    })
    df = df.sort_values(by='Support Count', ascending=False).head(10)
    print(df)

if __name__ == "__main__":
    file_path = 'Groceries_dataset.csv'
    min_support = 2
    max_itemset_size = 3

    frequent_itemsets = apriori_algorithm(min_support, file_path, max_itemset_size)
    transactions = get_transactions_from_file(file_path)
    rules = generate_rules(frequent_itemsets, transactions)

    visualize_frequent_itemsets(frequent_itemsets)
    print(f"Generated {len(rules)} association rules")
