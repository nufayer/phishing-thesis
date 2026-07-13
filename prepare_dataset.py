import pandas as pd

print("=" * 60)
print("PREPARING URL DATASET")
print("=" * 60)

# Load dataset
df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# Keep only URL and Label
df = df[["URL", "label"]]

print("\nOriginal Dataset :", len(df))

# Remove duplicate URLs
df = df.drop_duplicates(subset="URL")

print("After Removing Duplicates :", len(df))

# Save
df.to_csv("dataset/url_dataset.csv", index=False)

print("\nSaved Successfully!")
print("dataset/url_dataset.csv")