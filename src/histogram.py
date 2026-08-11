import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/dataset_train.csv")

def parse_data (df):
    return df.select_dtypes(include=['float64']).columns
    
def draw_all_in_grid(df):
    courses = parse_data(df)

    n_cols = 3
    n_rows = (len(courses) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    axes = axes.flatten()

    for i, course in enumerate(courses):
        sns.histplot(
            data=df,
            x=course,
            hue="Hogwarts House",
            bins=30,
            kde=True,
            ax=axes[i],
            stat="density",
            common_norm=False
        )
        axes[i].set_title(course)

    plt.tight_layout()
    plt.savefig("../histograms.png", dpi=150)
    print("Plot saved to ../histograms.png")
draw_all_in_grid(df)