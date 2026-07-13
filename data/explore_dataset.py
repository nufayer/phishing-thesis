import pandas as pd

# Load dataset
df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print(column)

print("\n")
print("=" * 70)
print("FIRST FIVE ROWS")
print("=" * 70)

print(df.head())

print("\n")
print("=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

print("\n")
print("=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())

print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(df["label"].value_counts())

print("\n")
print("=" * 70)
print("SUMMARY COMPLETE")
print("=" * 70)