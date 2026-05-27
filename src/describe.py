import sys
import pandas as pd
import math

class Describe:
    def __init__(self, df):
        self.df = df.select_dtypes(include=['number'])
        self.columns = self.df.columns
        self.stats_names = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Range", "IQR", "Var"]
        self.results = {}

    def get_count(self, values):
        i = 0
        for _ in values:
            i += 1
        return i

    def get_mean(self, values):
        total = self.get_count(values)
        if total == 0: return float('nan')
        sum = 0.0
        for i in values:
            sum += i
        return sum / total

    def get_std(self, values, mean):
        count = self.get_count(values)
        if count < 2: return 0.0
        sum_sq_diff = 0.0
        for x in values:
            sum_sq_diff += (x - mean) ** 2
        return math.sqrt(sum_sq_diff / (count - 1))

    def get_percentile(self, sorted_data, p):
        if not sorted_data: return float('nan')
        n = len(sorted_data)
        idx = p * (n - 1)
        low = int(idx)
        high = low + 1
        fraction = idx - low
        if high >= n:
            return float(sorted_data[low])
        return sorted_data[low] + (sorted_data[high] - sorted_data[low]) * fraction

    def calculate(self):
        for col in self.columns:
            raw_values = sorted([x for x in self.df[col] if pd.notna(x)])
            if not raw_values:
                self.results[col] = {
                    "Count": 0.0,
                    "Mean": float('nan'),
                    "Std": float('nan'),
                    "Min": float('nan'),
                    "25%": float('nan'),
                    "50%": float('nan'),
                    "75%": float('nan'),
                    "Max": float('nan'),
                    "Range": float('nan'),
                    "IQR": float('nan'),
                    "Var": float('nan')
                }
                continue
            mean = self.get_mean(raw_values)
            max_value = raw_values[-1]
            min_value = raw_values[0]
            std = self.get_std(raw_values, mean)
            q25 = self.get_percentile(raw_values, 0.25)
            q75 = self.get_percentile(raw_values, 0.75)
            self.results[col] = {
                "Count": self.get_count(raw_values),
                "Mean": mean,
                "Std": std,
                "Min": min_value,
                "25%": q25,
                "50%": self.get_percentile(raw_values, 0.50),
                "75%": q75,
                "Max": max_value,
                "Range" : max_value - min_value,
                "IQR" : q75 - q25,
                "Var": std **2
            }

    def display(self):
        if not self.results:
            return
        valid_columns = list(self.results.keys())
        col_widths = {col: max(len(col), 12) for col in valid_columns}
        header = f"{'':<10}"
        for col in valid_columns:
            header += f"{col:>{col_widths[col] + 2}}"
        print(header)
        for label in self.stats_names:
            row = f"{label:<10}"
            for col in valid_columns:
                val = self.results[col][label]
                row += f"{val:>{col_widths[col] + 2}.6f}"
            print(row)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python describe.py [dataset.csv]")
    else:
        try:
            df = pd.read_csv(sys.argv[1])
            desc = Describe(df)
            desc.calculate()
            desc.display()
        except Exception as e:
            print(f"Error: {e}")