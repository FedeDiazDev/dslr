import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../data/dataset_train.csv")


def pearson_correlation(x, y):
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        return 0.0

    mean_x = x.sum() / n
    mean_y = y.sum() / n

    cov = ((x - mean_x) * (y - mean_y)).sum() / (n - 1)
    std_x = np.sqrt(((x - mean_x) ** 2).sum() / (n - 1))
    std_y = np.sqrt(((y - mean_y) ** 2).sum() / (n - 1))

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)


def get_most_similar_features(df):
    courses = df.select_dtypes(include=['float64']).columns
    best_corr = -1.0
    best_pair = (None, None)
    tolerance = 1e-10

    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            r = abs(pearson_correlation(df[courses[i]], df[courses[j]]))
            if r - best_corr > tolerance:
                best_corr = r
                best_pair = (courses[i], courses[j])

    return best_pair[0], best_pair[1], best_corr


def draw_scatter_plot(df):
    feature1, feature2, corr_value = get_most_similar_features(df)
    print(f"Most similar features: {feature1} & {feature2}")
    print(f"Pearson correlation (|r|): {corr_value:.4f}")

    plt.figure(figsize=(10, 8))

    colors = {
        'Ravenclaw': '#1f77b4',
        'Slytherin': '#2ca02c',
        'Gryffindor': '#d62728',
        'Hufflepuff': '#ff7f0e'
    }

    for house in df['Hogwarts House'].dropna().unique():
        house_data = df[df['Hogwarts House'] == house]
        plt.scatter(
            house_data[feature1],
            house_data[feature2],
            alpha=0.5,
            label=house,
            color=colors.get(house, None),
            s=20
        )

    plt.xlabel(feature1, fontsize=12)
    plt.ylabel(feature2, fontsize=12)
    plt.title(
        f'Most Similar Features: {feature1} vs {feature2}\n'
        f'|Pearson r| = {corr_value:.4f} — High correlation means points align diagonally',
        fontsize=13
    )
    plt.legend(title='Hogwarts House', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(df[feature1].min() * 1.05, df[feature1].max() * 1.05)
    plt.ylim(df[feature2].min() * 1.05, df[feature2].max() * 1.05)
    plt.tight_layout()
    plt.savefig("../scatter_plot.png", dpi=150)
    print("Plot saved to ../scatter_plot.png")


draw_scatter_plot(df)
