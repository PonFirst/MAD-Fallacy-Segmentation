import pandas as pd

df = pd.read_csv("data/fallacy_first_version.csv")

grouped = df.groupby("Dialogue").size()
print("Fallacies per dialogue:")
print(grouped.describe())
print("\nDistribution:")
print(grouped.value_counts().sort_index())