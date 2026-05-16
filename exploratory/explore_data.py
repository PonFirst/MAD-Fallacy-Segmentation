import pandas as pd

df = pd.read_csv("data/fallacy_first_version.csv")

print("Shape:", df.shape)
print("\nFallacy distribution:")
print(df["Fallacy"].value_counts())

print("\n--- Sample row ---")
row = df.iloc[0]
print(f"Dialogue length (chars): {len(str(row['Dialogue']))}")
print(f"Snippet length (chars):  {len(str(row['Snippet']))}")
print(f"Fallacy: {row['Fallacy']}")
print(f"\nDialogue:\n{row['Dialogue']}")
print(f"\nSnippet:\n{row['Snippet']}")

print("\n--- Text length stats ---")
df["dialogue_len"] = df["Dialogue"].apply(lambda x: len(str(x).split()))
df["snippet_len"] = df["Snippet"].apply(lambda x: len(str(x).split()))
print("Dialogue word count:")
print(df["dialogue_len"].describe())
print("\nSnippet word count:")
print(df["snippet_len"].describe())

print("\n--- Snippet contained in Dialogue? ---")
df["span_found"] = df.apply(
    lambda r: str(r["Snippet"]).strip() in str(r["Dialogue"]), axis=1
)
print(df["span_found"].value_counts())