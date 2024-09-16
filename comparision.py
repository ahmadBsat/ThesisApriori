import time
import tracemalloc
import pandas as pd
from matplotlib import pyplot as plt
from mainApriori import apriori_algorithm
# from mainTRIE import apriori_with_trie

def compare_apriori_versions(file_path, min_support, min_confidence, max_itemset_size=3):
    # Results dictionary to store comparison metrics
    results = {
        'Version': [],
        'Execution Time (s)': [],
        'Memory Usage (MB)': [],
        'Frequent Itemsets': [],
    }

    # Helper function to measure execution time and memory usage
    def measure_performance(apriori_func, version_name):
        # Start measuring memory usage
        tracemalloc.start()

        # Start timing
        start_time = time.time()

        # Run the algorithm
        frequent_itemsets, loop_times = apriori_func(min_support, file_path, max_itemset_size)

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Measure memory usage
        current, peak = tracemalloc.get_traced_memory()
        memory_usage = peak / 10**6  # Convert bytes to MB

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
    frequent_itemsets_standard, loop_times_standard = measure_performance(apriori_algorithm, 'Standard Apriori')

    # Display comparison results
    comparison_df = pd.DataFrame(results)
    print(comparison_df)

    # Plotting execution times
    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df['Version'], comparison_df['Execution Time (s)'], color=['skyblue'])
    plt.xlabel('Version')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time Comparison')
    plt.show()

    # Plotting memory usage
    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df['Version'], comparison_df['Memory Usage (MB)'], color=['skyblue'])
    plt.xlabel('Version')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.show()

# Parameters
min_support = 2
min_confidence = 0.7
file_path = 'Groceries_dataset.csv'

# Run the comparison with only the standard algorithm
compare_apriori_versions(file_path, min_support, min_confidence)
