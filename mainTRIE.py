import os
import time
from itertools import combinations
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
from TrieClass import Trie
import tracemalloc
import psutil

# Function to load transactions from file
def get_transactions_from_file(file_path):
    transactions = []
    with open(file_path, 'r') as file:
        for line in file:
            items = line.strip().split()[1:]  # Skip transaction ID
            transactions.append(set(items))
    return transactions

# Optimized candidate generation with pruning
def generate_optimized_candidates(current_level_itemsets, trie):
    candidates = set()
    frequent_items = list(current_level_itemsets)
    len_frequent_items = len(frequent_items)

    for i in range(len_frequent_items):
        for j in range(i + 1, len_frequent_items):
            l1 = sorted(frequent_items[i])
            l2 = sorted(frequent_items[j])
            if l1[:-1] == l2[:-1]:  # Check prefix match
                candidate = frozenset(frequent_items[i] | frequent_items[j])
                if all(frozenset(candidate - {item}) in current_level_itemsets for item in candidate):
                    trie.insert(candidate)  # Insert into Trie for efficient lookup
                    candidates.add(candidate)
    return candidates

# Parallel support counting with transaction reduction
# Standalone function for multiprocessing
def count_chunk(chunk, candidates):
    counts = {candidate: 0 for candidate in candidates}
    for transaction in chunk:
        for candidate in candidates:
            if candidate.issubset(transaction):
                counts[candidate] += 1
    return counts

# Parallel support counting with transaction reduction
def parallel_support_counting(transactions, candidates, min_support, num_transactions):
    num_processes = min(4, os.cpu_count() // 2)  # Use fewer processes
    chunk_size = max(len(transactions) // num_processes, 1)
    transaction_chunks = [transactions[i:i + chunk_size] for i in range(0, len(transactions), chunk_size)]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(count_chunk, [(chunk, candidates) for chunk in transaction_chunks])

    total_counts = {}
    for result in results:
        for candidate, count in result.items():
            total_counts[candidate] = total_counts.get(candidate, 0) + count

    frequent_itemsets = {
        candidate: count / num_transactions
        for candidate, count in total_counts.items() if count / num_transactions >= min_support
    }

    # Reduce transactions by removing those that no longer contribute
    relevant_candidates = set().union(*frequent_itemsets.keys())
    reduced_transactions = [t.intersection(relevant_candidates) for t in transactions if t.intersection(relevant_candidates)]

    return frequent_itemsets, reduced_transactions

# Optimized Apriori algorithm using Trie and parallel support counting
def apriori_with_trie(transactions, min_support):
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions
    trie = Trie()
    frequent_itemsets = {}
    loop_count = 1
    current_level_itemsets = set()

    # Generate frequent 1-itemsets
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    for item, count in item_counts.items():
        if count >= min_support_count:
            candidate = frozenset([item])
            trie.insert(candidate)
            frequent_itemsets[candidate] = count / num_transactions
            current_level_itemsets.add(candidate)

    start_time = time.time()
    tracemalloc.start()
    start_memory_psutil = psutil.Process().memory_info().rss / (1024 * 1024)

    max_candidate_size = 6  # Limit candidate size

    while current_level_itemsets:
        print(f"Iteration {loop_count}: Generating candidates of size {loop_count + 1}")
        if loop_count >= max_candidate_size:
            print(f"Reached maximum candidate size {max_candidate_size}. Ending...")
            break

        # Generate candidates
        candidates = generate_optimized_candidates(current_level_itemsets, trie)
        if not candidates:
            print("No new candidates generated.")
            break

        # Count support and prune
        frequent_itemsets_k, transactions = parallel_support_counting(transactions, candidates, min_support, num_transactions)
        if not frequent_itemsets_k:
            print(f"No more frequent itemsets found in iteration {loop_count}. Ending...")
            break

        frequent_itemsets.update(frequent_itemsets_k)
        current_level_itemsets = set(frequent_itemsets_k.keys())
        loop_count += 1

        # Log iteration timing
        print(f"Iteration {loop_count} completed in {time.time() - start_time:.2f} seconds")

    elapsed_time = time.time() - start_time
    peak_memory_tracemalloc = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()
    end_memory_psutil = psutil.Process().memory_info().rss / (1024 * 1024)

    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Memory used (psutil): {end_memory_psutil - start_memory_psutil:.2f} MB")
    print(f"Peak memory (tracemalloc): {peak_memory_tracemalloc:.2f} MB")
    print(f"Number of frequent itemsets found: {len(frequent_itemsets)}")

    return frequent_itemsets, elapsed_time, (end_memory_psutil - start_memory_psutil), peak_memory_tracemalloc

# Visualize frequent itemsets
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

# Visualize performance metrics
def visualize_performance(combined_results):
    df = pd.DataFrame(combined_results)

    # Runtime comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['Dataset Name'], df['Runtime (seconds)'], color='orange')
    plt.xlabel('Dataset Name')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Memory usage comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['Dataset Name'], df['Memory Usage (MB) [psutil]'], color='blue', label='psutil')
    plt.bar(df['Dataset Name'], df['Peak Memory (MB) [tracemalloc]'], color='green', label='tracemalloc', alpha=0.7)
    plt.xlabel('Dataset Name')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Main function to process datasets in a folder
def process_datasets_in_folder(folder_path, min_support_ratio):
    datasets = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    combined_results = []
    all_frequent_itemsets = []  # Collect frequent itemsets for visualization later

    for dataset in datasets:
        file_path = os.path.join(folder_path, dataset)
        print(f"Processing dataset: {dataset}")
        transactions = get_transactions_from_file(file_path)
        frequent_itemsets, runtime, memory_psutil, memory_peak = apriori_with_trie(transactions, min_support_ratio)

        results = {
            "Dataset Name": dataset,
            "Runtime (seconds)": runtime,
            "Memory Usage (MB) [psutil]": memory_psutil,
            "Peak Memory (MB) [tracemalloc]": memory_peak,
            "Frequent Itemsets": len(frequent_itemsets),
        }
        combined_results.append(results)

        # Append for later visualization
        all_frequent_itemsets.append((dataset, frequent_itemsets))

    # Save combined results to a CSV file
    combined_results_df = pd.DataFrame(combined_results)
    combined_results_df.to_csv("Combined_Results.csv", index=False)
    print("Combined results saved to Combined_Results.csv")

    # Visualize results sequentially to avoid multiprocessing conflicts
    for dataset, frequent_itemsets in all_frequent_itemsets:
        print(f"Visualizing results for {dataset}")
        visualize_frequent_itemsets(frequent_itemsets)

    # Visualize performance metrics
    visualize_performance(combined_results)

if __name__ == "__main__":
    min_support_ratio = 0.05
    datasets_folder = 'TestDatasets'
    process_datasets_in_folder(datasets_folder, min_support_ratio)
