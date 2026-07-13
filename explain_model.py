import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import arff
from sklearn.model_selection import train_test_split

# =====================================================
# Load Dataset
# =====================================================

print("Loading Dataset...")

data, meta = arff.loadarff("dataset/Training Dataset.arff")

df = pd.DataFrame(data)

for col in df.columns:
    df[col] = df[col].apply(lambda x: int(x.decode("utf-8")))

df["Result"] = df["Result"].replace(-1, 0)

X = df.drop("Result", axis=1)
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset Ready!")

# =====================================================
# Load Best Model
# =====================================================

model = joblib.load("models/lightgbm.pkl")

print("Model Loaded!")

# =====================================================
# Create SHAP Explainer
# =====================================================

explainer = shap.TreeExplainer(model)

print("Generating SHAP values...")

shap_values = explainer.shap_values(X_test)

os.makedirs("figures", exist_ok=True)

# =====================================================
# SHAP Summary Plot
# =====================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    "figures/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP Summary Saved!")

print("\nDone!")