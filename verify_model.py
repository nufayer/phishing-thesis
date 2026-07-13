import pandas as pd

from predictor import predict_url

df = pd.read_csv("dataset/url_dataset.csv")

print("=" * 60)

# Test first 10 legitimate samples
print("\nTesting first 10 dataset samples\n")

for i in range(10):

    url = df.iloc[i]["URL"]
    label = df.iloc[i]["label"]

    result = predict_url(url)

    print(f"True Label : {label}")
    print(f"URL        : {url}")
    print(result)
    print("-" * 60)