import os

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT'))


def calculate_cdf(data, bins=50):
    """
    Calculate the cumulative distribution function (CDF) for a given dataset.

    Args:
        data (np.ndarray): The input dataset.
        bins (int, optional): The number of bins to use for histogram calculation. Defaults to 50.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the bin edges and the corresponding CDF values.
    """
    hist_data, bin_edges = np.histogram(data, bins=bins, density=True)
    cdf = np.cumsum(hist_data)
    cdf = cdf / cdf[-1]
    return bin_edges[1:], cdf


train = pd.read_csv(PROJECT_ROOT / 'data/training_data.csv')
features = train.columns[:-1]
feature_pairs = [(a, b) for a, b in combinations(features, 2)]

# Plot CDF for each feature in each pair
for feature_pair in feature_pairs:
    # Prepare the figure
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")

    for feature in feature_pair:
        x, y = calculate_cdf(train[feature])
        ax.plot(x, y, label=f'{feature} CDF')

    # Add labels and title
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Cumulative Density')
    ax.set_title('Cumulative Density Functions')
    ax.legend()

    # Show plot
    plt.show()
