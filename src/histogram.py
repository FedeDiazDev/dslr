import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/dataset_train.csv");

# def show_heatMap():
#     df = pd.read_csv("../data/dataset_train.csv");
#     courses = df.select_dtypes(include=['number']).drop(columns=['Index'])

#     plt.figure(figsize=(12, 10))
#     sns.heatmap(courses.corr(), annot=True, cmap='coolwarm', fmt=".2f")
#     plt.title("Correlation between Hogwarts Courses")
#     plt.show()

def parse_data ():
    courses = df.select_dtypes(include=['float64']).columns
    filter_courses = courses.drop('Defense Against the Dark Arts')
    return filter_courses
    
def draw_plot():
    fig = plt.figure(figsize=(20,15))
    fig.suptitle("House's Distribution")
    