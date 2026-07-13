import pandas as pd
from tqdm import tqdm

from html_features import extract_html_features

print("=" * 60)
print("GENERATING HTML FEATURES")
print("=" * 60)

df = pd.read_csv("dataset/url_dataset_v2.csv")

results = []

print(f"Total URLs: {len(df)}")
print()

for i, row in tqdm(df.iterrows(), total=len(df)):

    features = extract_html_features(row["URL"])

    if features is None:
        continue

    features["URL"] = row["URL"]
    features["label"] = row["label"]

    results.append(features)

    # Save every 100 successful websites
    if len(results) % 100 == 0:

        pd.DataFrame(results).to_csv(
            "dataset/html_features.csv",
            index=False
        )

        print(f"\nSaved {len(results)} websites")

# Final Save
pd.DataFrame(results).to_csv(
    "dataset/html_features.csv",
    index=False
)

print()
print("=" * 60)
print("Finished!")
print("Collected:", len(results))
print("=" * 60)