import matplotlib.pyplot as plt

# plot an histogram of all general metrics
def plot_metrics_histogram(metrics, dataset_name, log=False):
    plt.figure(figsize=(12, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.xticks(rotation=45)
    plt.ylabel("Metric Value")
    if log == True:
        plt.yscale("log")
    plt.title(f"Graph Metrics for {dataset_name} Graph")
    plt.show()