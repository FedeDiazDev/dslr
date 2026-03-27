
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df = pd.read_csv("../data/dataset_train.csv", index_col=0)
def draw_pair_plot(df):
    df = df.dropna()
    courses = df.select_dtypes(include=['float64']).columns
    sns.pairplot(
        df,
        vars=courses,
        hue="Hogwarts House",
        diag_kws={"multiple": "layer", "common_norm": False},
        plot_kws={'alpha': 0.5, 's': 10},
        height=1.5
    )

    plt.savefig("../pair_plot.png", dpi=100)
    print("Plot saved to ../pair_plot.png")

draw_pair_plot(df)
