import pandas as pd

from scipy.io import arff

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# ===================================
# LOAD DATASET
# ===================================

data, meta = arff.loadarff(
    "dataset/Training Dataset.arff"
)

df = pd.DataFrame(data)

for column in df.columns:
    df[column] = df[column].apply(
        lambda x: int(x.decode("utf-8"))
    )

# ===================================
# FEATURES
# ===================================

X = df.drop("Result", axis=1)

y = df["Result"]

# ===================================
# SPLIT
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ===================================
# MODELS
# ===================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "SVM":
        SVC(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

# ===================================
# RESULTS
# ===================================

print("\nMODEL COMPARISON\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"{name}: {accuracy*100:.2f}%"
    )