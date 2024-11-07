import time
import tracemalloc
import pandas as pd
from matplotlib import pyplot as plt
from mainApriori import apriori_algorithm
from mainTRIE import apriori_with_trie


def get_transactions_from_file(file_path):
    print("Reading the file...")
    df = pd.read_csv(file_path)

    # Exclude 'Item(s)' column if it's not needed
    item_columns = [col for col in df.columns if col.startswith('Item ')]
    print("Item columns identified:", item_columns)

    # Collect items from each row to form transactions
    transactions = []
    for idx, row in df.iterrows():
        # Get items from the item columns
        items = row[item_columns].dropna().astype(str).tolist()
        # Remove any empty strings or NaNs
        items = [item.strip() for item in items if item.strip() != '' and item.strip().lower() != 'nan']
        transactions.append(set(items))

    print("Number of transactions found:", len(transactions))
    return transactions


def compare_apriori_versions(transactions, min_support, min_confidence):
    # Results dictionary to store comparison metrics
    results = {
        'Version': [],
        'Execution Time (s)': [],
        'Memory Usage (MB)': [],
        'Frequent Itemsets': [],
    }

    # Helper function to measure execution time and memory usage
    def measure_performance(apriori_func, version_name, *args, **kwargs):
        # Clear previous memory traces
        tracemalloc.clear_traces()
        # Start measuring memory usage
        tracemalloc.start()

        # Start timing
        start_time = time.time()

        # Run the algorithm
        result = apriori_func(*args, **kwargs)

        # Handle case where only frequent_itemsets are returned
        if isinstance(result, tuple) and len(result) == 2:
            frequent_itemsets, loop_times = result
        else:
            frequent_itemsets = result
            loop_times = None

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Measure memory usage
        current, peak = tracemalloc.get_traced_memory()
        memory_usage = peak / 10 ** 6  # Convert bytes to MB

        # Stop measuring memory
        tracemalloc.stop()

        # Store results
        results['Version'].append(version_name)
        results['Execution Time (s)'].append(elapsed_time)
        results['Memory Usage (MB)'].append(memory_usage)
        results['Frequent Itemsets'].append(len(frequent_itemsets))

        return frequent_itemsets, loop_times
    # Run standard Apriori
    print("Running standard Apriori algorithm...")
    frequent_itemsets_standard, loop_times_standard = measure_performance(
        apriori_algorithm, 'Standard Apriori', transactions, min_support
    )

    # Run optimized Apriori with Trie
    print("Running optimized Apriori algorithm with Trie...")
    frequent_itemsets_trie, loop_times_trie = measure_performance(
        apriori_with_trie, 'Optimized Apriori with Trie', min_support, transactions
    )

    # Display comparison results
    comparison_df = pd.DataFrame(results)
    print(comparison_df)

    # Plotting execution times
    plt.figure(figsize=(10, 6))
    colors = ['skyblue', 'lightgreen']
    plt.bar(comparison_df['Version'], comparison_df['Execution Time (s)'], color=colors)
    plt.xlabel('Version')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time Comparison')
    plt.show()

    # Plotting memory usage
    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df['Version'], comparison_df['Memory Usage (MB)'], color=colors)
    plt.xlabel('Version')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.show()


if __name__ == "__main__":
    # Parameters
    min_support = 0.01  # Relative minimum support (e.g., 1%)
    min_confidence = 0.2
    file_path = 'groceries-groceries.csv'  # Update with your file path

    # Load transactions
    transactions = get_transactions_from_file(file_path)

    # Run the comparison
    compare_apriori_versions(transactions, min_support, min_confidence)
