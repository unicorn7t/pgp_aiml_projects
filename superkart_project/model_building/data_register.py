# Data registration: validate the dataset in the repo and print a summary.
import sys
import pandas as pd

DATA_PATH = "superkart_project/data/SuperKart.csv"

# Columns the data dictionary says must exist
EXPECTED_COLUMNS = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content",
    "Product_Allocated_Area", "Product_Type", "Product_MRP",
    "Store_Id", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Store_Sales_Total",
]

df = pd.read_csv(DATA_PATH)

# 1. Check for missing columns (fail loudly so the pipeline stops)
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    print("Validation FAILED. Missing columns:", missing)
    sys.exit(1)  # non-zero exit = GitHub Actions marks the job as failed

# 2. Print a short summary
print("Validation PASSED. All expected columns present.")
print("Shape          :", df.shape)
print("Missing values :", int(df.isnull().sum().sum()))
print("Duplicate rows :", int(df.duplicated().sum()))
print("\nColumn dtypes:\n", df.dtypes)
print("\nTarget summary (Product_Store_Sales_Total):\n",
      df["Product_Store_Sales_Total"].describe())
