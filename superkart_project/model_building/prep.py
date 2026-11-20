# Data preparation: clean, split, and save train/test files.
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "superkart_project/data/SuperKart.csv"
TARGET = "Product_Store_Sales_Total"

df = pd.read_csv(DATA_PATH)

# 1. Drop the unique product ID: it's a label, not a predictive feature
df = df.drop(columns=["Product_Id"])

# 2. Standardize inconsistent labels (e.g. 'reg' vs 'Regular'), if present
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
print("Sugar content values:\n", df["Product_Sugar_Content"].value_counts())

# 3. Remove duplicate rows
df = df.drop_duplicates()

# 4. Separate features (X) from target (y)
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 5. 80/20 split; random_state makes it reproducible
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Save in the working directory root (the workflow uploads these exact names)
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train:", Xtrain.shape, "| Test:", Xtest.shape)
