import pandas as pd

def read_data():
    data = pd.read_csv("./data/dataset_train.csv")
    data_numeric = data.select_dtypes(include=['float64'])
    # print (data)
    print(data_numeric.describe())
read_data()