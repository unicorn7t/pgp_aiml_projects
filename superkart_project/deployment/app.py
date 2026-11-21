import joblib
import pandas as pd
import streamlit as st

st.title("SuperKart Sales Forecast")

# Load the model and the store lookup table (one row per store, from the data)
model = joblib.load("superkart_project/deployment/best_model.joblib")
stores = (pd.read_csv("superkart_project/data/SuperKart.csv")
          .drop_duplicates("Store_Id")
          .set_index("Store_Id")[["Store_Size", "Store_Location_City_Type",
                                  "Store_Type", "Store_Establishment_Year"]])

st.subheader("Product details")
weight = st.number_input("Product weight", 0.0, 50.0, 12.0)
sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
area = st.number_input("Allocated display area ratio", 0.0, 1.0, 0.07)
ptype = st.selectbox("Product type", ["Fruits and Vegetables", "Snack Foods", "Frozen Foods",
    "Dairy", "Household", "Baking Goods", "Canned", "Health and Hygiene", "Meat",
    "Soft Drinks", "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"])
mrp = st.number_input("Product MRP", 0.0, 500.0, 150.0)

st.subheader("Store details")
store_id = st.selectbox("Store ID", sorted(stores.index))
info = stores.loc[store_id]   # the "vlookup": fetch this store's row
st.write(f"Size: {info['Store_Size']} | City: {info['Store_Location_City_Type']} | "
         f"Type: {info['Store_Type']} | Established: {info['Store_Establishment_Year']}")

# Column names must match the training data exactly
input_df = pd.DataFrame([{
    "Product_Weight": weight, "Product_Sugar_Content": sugar,
    "Product_Allocated_Area": area, "Product_Type": ptype, "Product_MRP": mrp,
    "Store_Id": store_id,
    "Store_Establishment_Year": int(info["Store_Establishment_Year"]),
    "Store_Size": info["Store_Size"],
    "Store_Location_City_Type": info["Store_Location_City_Type"],
    "Store_Type": info["Store_Type"]}])

if st.button("Predict"):
    st.write(input_df)
    st.success(f"Predicted sales: {model.predict(input_df)[0]:,.2f}")
