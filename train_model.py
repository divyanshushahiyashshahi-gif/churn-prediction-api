import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Simple dummy dataset
data = {
    "tenure": [1, 5, 10, 20, 3, 8, 15, 2],
    "monthly_charges": [50, 70, 80, 90, 60, 75, 85, 55],
    "contract_type": [0, 1, 1, 2, 0, 1, 2, 0],
    "churn": [1, 0, 0, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df[["tenure", "monthly_charges", "contract_type"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")