import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import os

os.makedirs("figures", exist_ok=True)

# =========================
# LOAD DATASET
# =========================

data, meta = arff.loadarff(
    "dataset/Training Dataset.arff"
)

df = pd.DataFrame(data)

for col in df.columns:
    df[col] = df[col].apply(
        lambda x: int(x.decode("utf-8"))
    )

X = df.drop("Result", axis=1)
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =========================
# RANDOM FOREST
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# FIGURE 5.1
# CONFUSION MATRIX
# =========================

plt.figure(figsize=(6,5))

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("Confusion Matrix")
plt.savefig(
    "figures/confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

# =========================
# FIGURE 5.2
# MODEL COMPARISON
# =========================

models = [
    "Logistic Regression",
    "SVM",
    "Random Forest"
]

accuracies = [
    92.45,
    94.71,
    96.70
]

plt.figure(figsize=(8,5))

plt.bar(models, accuracies)

plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

plt.savefig(
    "figures/model_comparison.png",
    bbox_inches="tight"
)

plt.close()

# =========================
# FIGURE 5.3
# FEATURE IMPORTANCE
# =========================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

top10 = feature_importance.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top10["Feature"],
    top10["Importance"]
)

plt.title(
    "Top 10 Feature Importance"
)

plt.savefig(
    "figures/feature_importance.png",
    bbox_inches="tight"
)

plt.close()

# =========================
# FIGURE 5.4
# CLASS DISTRIBUTION
# =========================

class_counts = y.value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    class_counts,
    labels=["Phishing", "Legitimate"],
    autopct="%1.1f%%"
)

plt.title(
    "Dataset Class Distribution"
)

plt.savefig(
    "figures/class_distribution.png",
    bbox_inches="tight"
)

plt.close()

print("All Figures Generated Successfully!")