import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, get_train_test_split

df = load_dataset()
train, test = get_train_test_split(df)

test["annotation_count"] = test["annotations"].apply(len)
test_sorted = test.sort_values("annotation_count", ascending=False).reset_index(drop=True)

print("\nTop 10 dialogues by fallacy count in test set:\n")
for i in range(min(10, len(test_sorted))):
    row = test_sorted.iloc[i]
    print(f"Rank {i+1} — {row['annotation_count']} annotations")
    print(f"  Dialogue length: {len(row['dialogue'].split())} words")
    print(f"  Fallacies:")
    for ann in row["annotations"]:
        print(f"    - {ann['fallacy']}: {ann['snippet']}")
    print()

print(f"Highest: {test_sorted.iloc[0]['annotation_count']} annotations")
print(f"Lowest:  {test_sorted.iloc[-1]['annotation_count']} annotations")
print(f"Average: {test['annotation_count'].mean():.1f} annotations")
print(f"Median:  {test['annotation_count'].median():.1f} annotations")

# full dataset duplicate check
total_dupes = 0
for _, row in test.iterrows():
    seen = set()
    for ann in row["annotations"]:
        key = (ann["span_start"], ann["span_end"])
        if key in seen:
            total_dupes += 1
        seen.add(key)

print(f"\nRemaining duplicate spans in test after dedup: {total_dupes}")
print(f"Total test annotations after dedup: {test['annotation_count'].sum()}")