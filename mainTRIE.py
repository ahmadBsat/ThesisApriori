import time
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool
from TrieClass import Trie
import psutil  # To monitor CPU and memory usage
import logging

# Configure logging to write to a file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("enhancedPerf.txt"),
                        logging.StreamHandler()
                    ])


def log_system_performance(stage):
    """
    Logs the system's CPU and memory usage at different stages of the algorithm.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    logging.info(f"{stage} - CPU Usage: {cpu_usage}%, Memory Usage: {memory_info.percent}%")


def get_transactions_from_file(file_path, sample_size=None):
    """
    Reads the dataset, handles missing values, and converts each transaction into a list of items.
    Optionally uses a subset of the dataset if sample_size is provided.
    """
    logging.info("Reading transactions from file...")
    df = pd.read_csv(file_path)

    if sample_size:
        # Take a sample of the dataset if a sample_size is provided
        df = df.sample(n=sample_size, random_state=1)  # Sample a subset for testing
        logging.info(f"Using a sample size of {sample_size} transactions for testing.")

    item_columns = [col for col in df.columns if col.startswith('Item ')]
    transactions = []
    for idx, row in df.iterrows():
        items = row[item_columns].dropna().astype(str).tolist()
        items = [item.strip() for item in items if item.strip()]
        transactions.append(set(items))

    logging.info(f"Number of transactions: {len(transactions)}")
    return transactions


def count_support_parallel(transactions, candidates):
    count_dict = {candidate: 0 for candidate in candidates}
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                count_dict[candidate] += 1
    return count_dict


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


def prune_candidates(trie, candidates, min_support, num_transactions):
    frequent_itemsets = {}
    for candidate in candidates:
        support_count = trie.get_support(candidate)
        if support_count >= min_support * num_transactions:
            support = support_count / num_transactions
            frequent_itemsets[candidate] = support
    return frequent_itemsets


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
    log_system_performance("Before Apriori Execution")

    # Iteratively generate and prune itemsets of increasing size
    while current_level_itemsets:
        logging.info(f"Iteration {loop_count}: Generating candidates of size {loop_count + 1}")
        # Generate candidates of size k+1 from frequent k-itemsets
        candidates = trie.generate_candidates(current_level_itemsets)
        if not candidates:
            logging.info("No new candidates generated.")
            break

        # Count supports of candidates using parallel processing
        frequent_itemsets_k = parallel_support_counting(transactions, candidates, min_support, num_transactions)

        if not frequent_itemsets_k:
            logging.info(f"No more frequent itemsets found in iteration {loop_count}. Ending...")
            break

        # Update frequent itemsets
        frequent_itemsets.update(frequent_itemsets_k)
        current_level_itemsets = set(frequent_itemsets_k.keys())
        loop_count += 1
        log_system_performance(f"After Iteration {loop_count}")

    # End timing
    elapsed_time = time.time() - start_time
    logging.info(f"Total time taken: {elapsed_time:.2f} seconds")
    log_system_performance("After Apriori Completion")
    logging.info(f"Number of frequent itemsets found: {len(frequent_itemsets)}")

    return frequent_itemsets


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
    logging.info(f"Generated {len(rules)} association rules")
    return rules


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


def save_frequent_itemsets_to_csv(frequent_itemsets, file_name='frequent_itemsets.csv'):
    """
    Saves the frequent itemsets (or groups of items) to a CSV file so you can analyze them later.
    """
    itemsets = [' & '.join(itemset) for itemset in frequent_itemsets.keys()]  # Join items in itemset with '&'
    supports = list(frequent_itemsets.values())  # Get the support values of the itemsets

    # Create a DataFrame with itemsets and their corresponding support
    df = pd.DataFrame({
        'Itemset': itemsets,
        'Support': supports
    })

    # Save the DataFrame to a CSV file
    df.to_csv(file_name, index=False)
    logging.info(f"Frequent itemsets saved to {file_name}")


def save_rules_to_csv(rules, file_name='association_rules.csv'):
    """
    Saves the generated association rules (Antecedent, Consequent, Support, Confidence, Lift) to a CSV file.
    """
    # Create a DataFrame with rules
    df = pd.DataFrame(rules)

    # Save the DataFrame to a CSV file
    df.to_csv(file_name, index=False)
    logging.info(f"Association rules saved to {file_name}")

def prepare_transaction_samples_three_items():
    """
    Prepares predefined transaction samples including transactions with three items.
    Returns a dictionary with sample names and transactions.
    """
    samples = {
        "Sample 1": [
            {"A", "B", "C"},
            {"A", "C", "D"},
            {"B", "C", "D"},
            {"A", "B", "D"}
        ],
        "Sample 2": [
            {"A", "B", "C"},
            {"A", "C"},
            {"B", "C", "D"},
            {"A", "B"}
        ],
        "Sample 3": [
            {"A", "B", "C"},
            {"A", "B", "D"},
            {"B", "C", "D"},
            {"A", "C"}
        ]
    }
    return samples

def analyze_samples(samples, min_support_ratio, min_confidence):
    """
    Analyzes each sample using the Apriori with Trie algorithm.
    Saves results and association rules for each sample.
    """
    results = {}
    for sample_name, transactions in samples.items():
        print(f"Analyzing {sample_name}...")

        # Run Apriori algorithm
        frequent_itemsets = apriori_with_trie(transactions, min_support_ratio)

        # Generate association rules
        rules = generate_rules(frequent_itemsets, min_confidence)

        # Save results
        frequent_file = f"{sample_name.replace(' ', '_').lower()}_frequent_itemsets.csv"
        rules_file = f"{sample_name.replace(' ', '_').lower()}_rules.csv"
        save_frequent_itemsets_to_csv(frequent_itemsets, frequent_file)
        save_rules_to_csv(rules, rules_file)

        # Store results for document creation
        results[sample_name] = {
            "transactions": transactions,
            "frequent_itemsets": frequent_itemsets,
            "rules": rules
        }

    return results

def main():
    # Parameters
    min_support_ratio = 0.5  # Minimum 50% support
    min_confidence = 0.6  # Minimum 60% confidence

    # Step 1: Prepare transaction samples
    samples = prepare_transaction_samples_three_items()

    # Step 2: Analyze samples
    results = analyze_samples(samples, min_support_ratio, min_confidence)

    # Print summary
    for sample_name, data in results.items():
        print(f"\n=== {sample_name} ===")
        print("Frequent Itemsets:")
        for itemset, support in data["frequent_itemsets"].items():
            print(f"  {set(itemset)}: {support:.2f}")
        print("\nAssociation Rules:")
        for rule in data["rules"]:
            print(f"  {rule['Antecedent']} -> {rule['Consequent']}: "
                  f"Support: {rule['Support']:.2f}, "
                  f"Confidence: {rule['Confidence']:.2f}, "
                  f"Lift: {rule['Lift']:.2f}")

if __name__ == "__main__":
    main()

