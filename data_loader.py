import pandas as pd
import re
from sklearn.model_selection import train_test_split


def load_dataset(path="data/fallacy_first_version.csv"):
    df = pd.read_csv(path)
    df["Fallacy"] = df["Fallacy"].str.strip()

    def find_span(row):
        dialogue = str(row["Dialogue"])
        snippet = str(row["Snippet"]).strip()
        idx = dialogue.find(snippet)
        if idx != -1:
            return idx, idx + len(snippet), True
        snippet_stripped = re.sub(r'\s+', ' ', snippet).strip()
        dialogue_normalized = re.sub(r'\s+', ' ', dialogue)
        idx = dialogue_normalized.find(snippet_stripped)
        if idx != -1:
            return idx, idx + len(snippet_stripped), True
        return -1, -1, False

    spans = df.apply(find_span, axis=1)
    df["span_start"] = spans.apply(lambda x: x[0])
    df["span_end"] = spans.apply(lambda x: x[1])
    df["span_found"] = spans.apply(lambda x: x[2])

    total = len(df)
    found = df["span_found"].sum()
    print(f"Loaded {total} annotations across dialogues")
    print(f"Span found: {found} ({100*found/total:.1f}%)")
    print(f"\nFallacy distribution:")
    print(df["Fallacy"].value_counts())

    return df


def get_train_test_split(df, test_size=0.2, seed=42):
    """
    Split at the dialogue level not the annotation level.
    Deduplicates annotations with identical span coordinates.
    """
    records = []
    total_dupes = 0

    for dialogue_text, group in df.groupby("Dialogue", sort=False):
        valid = group[group["span_found"]]
        if len(valid) == 0:
            continue

        annotations = []
        seen_spans = set()

        for _, row in valid.iterrows():
            key = (int(row["span_start"]), int(row["span_end"]))
            if key in seen_spans:
                total_dupes += 1
                continue
            seen_spans.add(key)
            annotations.append({
                "fallacy": row["Fallacy"],
                "snippet": row["Snippet"],
                "span_start": int(row["span_start"]),
                "span_end": int(row["span_end"])
            })

        records.append({
            "dialogue": dialogue_text,
            "annotations": annotations
        })

    print(f"\nDuplicate spans removed: {total_dupes}")

    dialogue_groups = pd.DataFrame(records)

    train, test = train_test_split(
        dialogue_groups,
        test_size=test_size,
        random_state=seed
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_ann = sum(len(x) for x in train["annotations"])
    test_ann = sum(len(x) for x in test["annotations"])

    print(f"\nTrain dialogues: {len(train)}")
    print(f"Test dialogues: {len(test)}")
    print(f"Train annotations: {train_ann}")
    print(f"Test annotations: {test_ann}")

    return train, test


if __name__ == "__main__":
    df = load_dataset()
    train, test = get_train_test_split(df)

    print("\nSample test dialogue:")
    sample = test.iloc[0]
    print(f"  Dialogue length: {len(sample['dialogue'].split())} words")
    print(f"  Gold annotations: {len(sample['annotations'])}")
    for ann in sample["annotations"]:
        print(f"    - {ann['fallacy']}: {ann['snippet'][:60]}...")