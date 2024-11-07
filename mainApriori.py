import pandas as pd
from itertools import combinations
from collections import Counter
import logging

# Configure logging to write to both a file and the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("apriori_logs.txt"),
                        logging.StreamHandler()
                    ])


def get_transactions_from_file(file_path):
    """
    Reads the dataset, handles missing values, and converts each transaction into a list of items.
    """
    logging.info("Reading transactions from file...")
    df = pd.read_csv(file_path)
    transactions = df.apply(lambda row: row.dropna().astype(str).tolist(), axis=1).tolist()
    logging.info(f"Number of transactions: {len(transactions)}")
    return transactions


def get_all_itemsets(items, length):
    """
    Generate all possible itemsets of a given length (brute force, no pruning).
    """
    return [frozenset(itemset) for itemset in combinations(items, length)]


def count_support(transactions, itemsets):
    """
    Efficient counting of support for itemsets using Counter.
    """
    logging.info(f"Counting support for {len(itemsets)} itemsets...")
    itemset_counts = Counter()
    for transaction in transactions:
        for itemset in itemsets:
            if itemset.issubset(transaction):  # Check if the itemset is contained in the transaction
                itemset_counts[itemset] += 1
    return itemset_counts


def apriori_brute_force(transactions, min_support):
    """
    Faster version of the Apriori algorithm with some optimizations.
    """
    items = set(item for transaction in transactions for item in transaction)  # All unique items in dataset
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions  # Support count threshold

    logging.info(f"Running Apriori with minimum support of {min_support}...")

    frequent_itemsets = {}
    k = 1

    while True:
        # Generate all possible itemsets of size k
        itemsets = get_all_itemsets(items, k)
        if not itemsets:
            break

        # Count the support of all itemsets
        itemset_counts = count_support(transactions, itemsets)

        current_frequent_itemsets = {
            itemset: count / num_transactions
            for itemset, count in itemset_counts.items()
            if count >= min_support_count
        }

        if not current_frequent_itemsets:
            logging.info(f"No frequent itemsets found for size {k}. Stopping...")
            break  # Stop if no frequent itemsets are found for this size

        frequent_itemsets.update(current_frequent_itemsets)
        logging.info(f"Found {len(current_frequent_itemsets)} frequent itemsets of size {k}.")
        k += 1  # Increase the size of itemsets for the next iteration

    logging.info(f"Total frequent itemsets found: {len(frequent_itemsets)}")
    return frequent_itemsets


def generate_rules_brute_force(frequent_itemsets, min_confidence):
    """
    Generates association rules without optimization: brute force all combinations of frequent itemsets.
    """
    logging.info("Generating association rules...")
    rules = []

    for itemset in frequent_itemsets:
        if len(itemset) >= 2:
            subsets = [frozenset(x) for x in combinations(itemset, r) for r in range(1, len(itemset))]
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

    logging.info(f"Generated {len(rules)} association rules.")
    return rules


def visualize_frequent_itemsets(frequent_itemsets):
    """
    Visualizes the top 10 frequent itemsets by their support values.
    """
    import matplotlib.pyplot as plt

    df = pd.DataFrame({
        'Itemset': [' & '.join(itemset) for itemset in frequent_itemsets.keys()],
        'Support': list(frequent_itemsets.values())
    })

    df = df.sort_values(by='Support', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(df['Itemset'], df['Support'], color='skyblue')
    plt.xlabel('Support')
    plt.ylabel('Itemset')
    plt.title('Top 10 Frequent Itemsets')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    # Define the file path, minimum support, and confidence
    file_path = 'groceries-groceries.csv'  # Update with the correct file path
    min_support = 0.01  # Minimum support as a fraction (e.g., 1%)
    min_confidence = 0.1  # Minimum confidence threshold

    logging.info("Starting Apriori algorithm...")

    # Get transactions from file
    transactions = get_transactions_from_file(file_path)

    # Run the brute-force Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori_brute_force(transactions, min_support)

    # Generate association rules from the frequent itemsets
    rules = generate_rules_brute_force(frequent_itemsets, min_confidence)

    # Visualize the top 10 frequent itemsets
    visualize_frequent_itemsets(frequent_itemsets)

    # Convert rules into a DataFrame and save them to an Excel file
    rules_df = pd.DataFrame(rules)
    output_file = 'association_rules.xlsx'
    rules_df.to_excel(output_file, index=False)
    logging.info(f"Association rules have been saved to {output_file}")

    # Optionally, display the generated rules
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    logging.info("Displaying generated rules...")
    print(rules_df)
