import pandas as pd
import lightgbm as lgb
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

print("=" * 70)
print("BASELINE LIGHTGBM MODEL")
print("=" * 70)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

print("\nLoading Dataset...")

df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

print("Dataset Loaded Successfully!")

# ---------------------------------------------------------
# Remove NLP Columns
# ---------------------------------------------------------

print("\nRemoving text columns...")

text_columns = [
    "URL",
    "Domain",
    "Title",
    "TLD"
]

X = df.drop(columns=text_columns + ["label"])

y = df["label"]

print("Remaining Features :", X.shape[1])

# ---------------------------------------------------------
# Train Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ---------------------------------------------------------
# LightGBM
# ---------------------------------------------------------

print("\nTraining LightGBM...")

model = lgb.LGBMClassifier(

    n_estimators=300,

    learning_rate=0.05,

    num_leaves=63,

    max_depth=8,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42

)

model.fit(X_train, y_train)

print("Training Complete!")

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

pred = model.predict(X_test)

prob = model.predict_proba(X_test)[:,1]

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"Accuracy  : {accuracy_score(y_test,pred):.4f}")
print(f"Precision : {precision_score(y_test,pred):.4f}")
print(f"Recall    : {recall_score(y_test,pred):.4f}")
print(f"F1 Score  : {f1_score(y_test,pred):.4f}")
print(f"ROC AUC   : {roc_auc_score(y_test,prob):.4f}")

print("\nConfusion Matrix")

print(confusion_matrix(y_test,pred))

print("\nClassification Report\n")

print(classification_report(y_test,pred))

# ---------------------------------------------------------
# Save Model
# ---------------------------------------------------------

joblib.dump(model,"models/baseline_lightgbm.pkl")

print("\nModel Saved!")

# ---------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Most Important Features\n")
print(importance.head(20))

importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

print("Feature Importance Saved!")

# ---------------------------------------------------------
# SHAP
# ---------------------------------------------------------

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)

sample = X_test.sample(1000, random_state=42)

shap_values = explainer.shap_values(sample)

plt.figure(figsize=(10,8))

shap.summary_plot(

    shap_values,

    sample,

    show=False

)

plt.tight_layout()

plt.savefig(

    "results/shap_summary.png",

    dpi=300

)

print("SHAP Summary Saved!")

print("\nDone.")