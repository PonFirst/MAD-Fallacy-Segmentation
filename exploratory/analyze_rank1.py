import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, get_train_test_split

df = load_dataset()
train, test = get_train_test_split(df)

test["annotation_count"] = test["annotations"].apply(len)
test_sorted = test.sort_values("annotation_count", ascending=False).reset_index(drop=True)

rank1 = test_sorted.iloc[0]

print(f"Rank 1 Dialogue")
print(f"Annotation count: {rank1['annotation_count']}")
print(f"Dialogue length: {len(rank1['dialogue'].split())} words")
print(f"\nFull dialogue:\n{rank1['dialogue']}")
print(f"\n{'='*60}")
print(f"Gold annotations ({rank1['annotation_count']}):\n")

for i, ann in enumerate(rank1["annotations"]):
    print(f"[{i+1}] {ann['fallacy']}")
    print(f"     Span: [{ann['span_start']}:{ann['span_end']}]")
    print(f"     Snippet: {ann['snippet']}")
    print()