import pandas as pd
from feature_extractor import extract_features

print("=" * 70)
print("GENERATING FEATURES")
print("=" * 70)

# Load URL dataset
df = pd.read_csv("dataset/url_dataset.csv")

print(f"\nTotal URLs : {len(df)}")

feature_rows = []

print("\nExtracting Features...")

for index, row in df.iterrows():

    features = extract_features(row["URL"])

    features["label"] = row["label"]

    feature_rows.append(features)

    if (index + 1) % 10000 == 0:
        print(f"Processed {index+1:,} URLs")

# Create DataFrame
feature_df = pd.DataFrame(feature_rows)

print("\nFeature Extraction Complete!")

print(feature_df.head())

# Save Dataset
feature_df.to_csv(
    "dataset/generated_features.csv",
    index=False
)

print("\nSaved Successfully!")

print("dataset/generated_features.csv")