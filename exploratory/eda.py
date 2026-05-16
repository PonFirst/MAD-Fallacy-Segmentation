import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/fallacy_first_version.csv")

# get unique dialogues with their fallacy counts
dialogue_groups = df.groupby("Dialogue").agg(
    fallacy_count=("Fallacy", "count"),
    fallacies=("Fallacy", list),
    snippets=("Snippet", list)
).reset_index()

print(f"Total unique dialogues: {len(dialogue_groups)}")
print(f"Average fallacies per dialogue: {dialogue_groups['fallacy_count'].mean():.1f}")

# split at dialogue level not sample level
train_dialogues, test_dialogues = train_test_split(
    dialogue_groups,
    test_size=0.2,
    random_state=42
)

print(f"\nTrain dialogues: {len(train_dialogues)}")
print(f"Test dialogues: {len(test_dialogues)}")
print(f"Train annotations: {train_dialogues['fallacy_count'].sum()}")
print(f"Test annotations: {test_dialogues['fallacy_count'].sum()}")