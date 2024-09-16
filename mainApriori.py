import time
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain, combinations

def get_transactions_from_file(file_path):
    print("Reading the file...")
    df = pd.read_csv(file_path)
    grouped = df.groupby('Member_number')['itemDescription'].apply(list)
    transactions = grouped.tolist()
    print("Number of transactions found:", len(transactions))
    return transactions


def get_candidates(transaction_list, length):
    print(f"Generating candidates of length {length}...")
    candidates = []
    for transaction in transaction_list:
        for combo in combinations(transaction, length):
            candidates.append(combo)
    return list(set(candidates))


def count_support(candidates, transaction_list):
    print("Counting support for candidates...")
    support_count = {}
    for candidate in candidates:
        support_count[candidate] = sum(
            [1 for transaction in transaction_list if set(candidate).issubset(set(transaction))])
    return support_count


def prune_candidates(support_count, min_support):
    print(f"Pruning candidates with support < {min_support}...")
    return {k: v for k, v in support_count.items() if v >= min_support}


def apriori_algorithm(min_support, file_path, max_itemset_size=3):
    global transaction_list  # To make transaction_list accessible in generate_rules
    transactions = get_transactions_from_file(file_path)
    transaction_list = list(map(set, transactions))

    k = 1
    frequent_itemsets = {}
    loop_count = 1
    loop_times = []  # To store the time taken for each loop

    # Start timing
    start_time = time.time()

    while k <= max_itemset_size:
        print(f"Starting loop {loop_count} with itemset size {k}...")
        loop_start_time = time.time()  # Start timing for the loop

        candidates = get_candidates(transaction_list, k)
        print(f"Generated {len(candidates)} candidates of length {k}")
        support_count = count_support(candidates, transaction_list)
        frequent_k_itemsets = prune_candidates(support_count, min_support)

        loop_end_time = time.time()  # End timing for the loop
        loop_time = loop_end_time - loop_start_time
        loop_times.append(loop_time)  # Store the loop time

        if not frequent_k_itemsets:
            print(f"No more frequent itemsets found in loop {loop_count}. Ending...")
            break

        frequent_itemsets.update(frequent_k_itemsets)
        k += 1
        loop_count += 1

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(loop_times)
    print(f"Time taken: {elapsed_time:.2f} seconds")
    return frequent_itemsets, loop_times


def generate_rules(frequent_itemsets, transaction_list, min_confidence=0.6):
    rules = []

    for itemset in frequent_itemsets.keys():
        length = len(itemset)
        if length > 1:  # Rules can only be generated for itemsets of size 2 or more
            subsets = chain(*[combinations(itemset, i) for i in range(1, length)])
            for antecedent in subsets:
                antecedent = frozenset(antecedent)
                consequent = frozenset(itemset) - antecedent

                # Check if both antecedent and consequent are in frequent itemsets
                if antecedent in frequent_itemsets and consequent in frequent_itemsets:
                    confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                    support = frequent_itemsets[itemset] / len(transaction_list)
                    lift = confidence / (frequent_itemsets[consequent] / len(transaction_list))

                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, support, confidence, lift))

    return rules



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


def save_frequent_itemsets_to_csv(frequent_itemsets, file_name='frequent_itemsets.csv'):
    itemsets = [' & '.join(itemset) for itemset in frequent_itemsets.keys()]
    support_counts = list(frequent_itemsets.values())
    df = pd.DataFrame({'Itemset': itemsets, 'Support Count': support_counts})
    df = df.sort_values(by='Support Count', ascending=False)  # Sort by support count
    df.to_csv(file_name, index=False)
    print(f"Frequent itemsets saved to {file_name}")


def save_rules_to_csv(rules, file_name='association_rules.csv'):
    df = pd.DataFrame(rules, columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift'])
    df['Antecedent'] = df['Antecedent'].apply(lambda x: ', '.join(list(x)))
    df['Consequent'] = df['Consequent'].apply(lambda x: ', '.join(list(x)))
    df.to_csv(file_name, index=False)
    print(f"Association rules saved to {file_name}")


# Ensure this block only runs when this file is executed directly
if __name__ == "__main__":
    # Set a minimum support and confidence threshold
    min_support = 2
    min_confidence = 0.7

    # Provide the correct file path
    file_path = 'Groceries_dataset.csv'

    # Run the Apriori algorithm with the file data
    print("Running the Apriori algorithm...")
    frequent_itemsets, loop_times = apriori_algorithm(min_support, file_path)

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

