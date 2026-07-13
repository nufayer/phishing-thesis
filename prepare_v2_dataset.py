import pandas as pd

print("=" * 60)
print("PREPARING VERSION 2 DATASET")
print("=" * 60)

df = pd.read_csv("dataset/url_dataset.csv")

print("Original:", len(df))

# Separate classes
legitimate = df[df["label"] == 1]
phishing = df[df["label"] == 0]

print("Original Legitimate:", len(legitimate))
print("Original Phishing:", len(phishing))

# Only use 1000 from each class
legitimate = legitimate.sample(
    n=1000,
    random_state=42
)

phishing = phishing.sample(
    n=1000,
    random_state=42
)

dataset = pd.concat([
    legitimate,
    phishing
])

dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print()
print(dataset["label"].value_counts())

dataset.to_csv(
    "dataset/url_dataset_v2.csv",
    index=False
)

print()
print("Saved Successfully!")
print("dataset/url_dataset_v2.csv")