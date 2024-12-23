import pandas as pd
import psutil  # To monitor CPU and memory usage
import logging
from itertools import combinations
from collections import Counter
import time

# Configure logging to write to a file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("originalPerf.txt"),
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
        df = df.sample(n=sample_size, random_state=1)  # Sample a subset for testing
        logging.info(f"Using a sample size of {sample_size} transactions for testing.")
    transactions = df.apply(lambda row: row.dropna().astype(str).tolist(), axis=1).tolist()
    logging.info(f"Number of transactions: {len(transactions)}")
    return transactions


def get_all_itemsets(items, length):
    """
    Generate all possible itemsets of a given length (brute force, no pruning).
    """
    logging.info(f"Generating all possible itemsets of length {length}...")
    return [frozenset(itemset) for itemset in combinations(items, length)]


def count_support(transactions, itemsets):
    """
    Efficient counting of support for itemsets using Counter.
    """
    logging.info(f"Counting support for {len(itemsets)} itemsets...")
    itemset_counts = Counter()
    for transaction in transactions:
        for itemset in itemsets:
            if itemset.issubset(transaction):
                itemset_counts[itemset] += 1
    logging.info(f"Finished counting support for itemsets.")
    return itemset_counts


def apriori_brute_force(transactions, min_support, max_itemset_size=3):
    """
    Basic version of the Apriori algorithm with a maximum itemset size limit.
    """
    items = set(item for transaction in transactions for item in transaction)
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions

    logging.info(f"Running Apriori with minimum support of {min_support} and max itemset size of {max_itemset_size}...")
    log_system_performance("Before Apriori Execution")

    frequent_itemsets = {}
    k = 1

    while k <= max_itemset_size:
        # Generate all possible itemsets of size k
        itemsets = get_all_itemsets(items, k)
        if not itemsets:
            break

        # Count the support of all itemsets
        start_time = time.time()
        itemset_counts = count_support(transactions, itemsets)
        elapsed_time = time.time() - start_time
        logging.info(f"Support counting for itemsets of size {k} took {elapsed_time:.2f} seconds.")

        current_frequent_itemsets = {
            itemset: count / num_transactions
            for itemset, count in itemset_counts.items()
            if count >= min_support_count
        }

        if not current_frequent_itemsets:
            logging.info(f"No frequent itemsets found for size {k}. Stopping...")
            break

        frequent_itemsets.update(current_frequent_itemsets)
        logging.info(f"Iteration {k}: Found {len(current_frequent_itemsets)} frequent itemsets of size {k}. ")
        log_system_performance(f"After Iteration {k}")

        k += 1

    logging.info(f"Total frequent itemsets found: {len(frequent_itemsets)}")
    log_system_performance("After Apriori Completion")
    return frequent_itemsets


def generate_rules_brute_force(frequent_itemsets, min_confidence):
    """
    Generates association rules without optimization: brute force all combinations of frequent itemsets.
    """
    logging.info("Generating association rules...")
    start_time = time.time()
    rules = []

    for itemset in frequent_itemsets:
        if len(itemset) >= 2:
            # Generate all non-empty subsets of the itemset
            subsets = [frozenset(x) for r in range(1, len(itemset)) for x in combinations(itemset, r)]
            for antecedent in subsets:
                consequent = itemset - antecedent
                if consequent:
                    antecedent_support = frequent_itemsets.get(antecedent, 0)
                    if antecedent_support > 0:
                        confidence = frequent_itemsets[itemset] / antecedent_support
                        if confidence >= min_confidence:
                            consequent_support = frequent_itemsets.get(consequent, 0)
                            lift = confidence / consequent_support if consequent_support > 0 else 0
                            support = frequent_itemsets[itemset]
                            rules.append({
                                'Antecedent': ', '.join(antecedent),
                                'Consequent': ', '.join(consequent),
                                'Support': support,
                                'Confidence': confidence,
                                'Lift': lift
                            })

    elapsed_time = time.time() - start_time
    logging.info(f"Generated {len(rules)} association rules in {elapsed_time:.2f} seconds.")
    log_system_performance("After Rule Generation")
    return rules


if __name__ == "__main__":
    # Define the file path, minimum support, and confidence
    file_path = 'groceries-groceries.csv'  # Update with the correct file path
    min_support = 0.01  # Minimum support as a fraction (e.g., 1%)
    min_confidence = 0.1  # Minimum confidence threshold
    sample_size = 1000  # Adjust sample size for quicker testing if needed
    max_itemset_size = 3  # Limit maximum itemset size to avoid excessive computation

    logging.info("Starting Apriori algorithm...")
    log_system_performance("Initial System Performance")
    transactions = get_transactions_from_file(file_path, sample_size=sample_size)

    # Run the brute-force Apriori algorithm to find frequent itemsets with a max itemset size limit
    frequent_itemsets = apriori_brute_force(transactions, min_support, max_itemset_size=max_itemset_size)

    # Generate association rules from the frequent itemsets
    rules = generate_rules_brute_force(frequent_itemsets, min_confidence)

    logging.info("Apriori algorithm completed.")
    log_system_performance("Final System Performance")