import joblib
import pandas as pd
import streamlit as st

st.title("SuperKart Sales Forecast")

# Load the model that sits in the same folder as this file
model = joblib.load("superkart_project/deployment/best_model.joblib")

st.subheader("Product details")
weight = st.number_input("Product weight", 0.0, 50.0, 12.0)
sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
area = st.number_input("Allocated display area ratio", 0.0, 1.0, 0.07)
ptype = st.selectbox("Product type", ["Fruits and Vegetables", "Snack Foods", "Frozen Foods",
    "Dairy", "Household", "Baking Goods", "Canned", "Health and Hygiene", "Meat",
    "Soft Drinks", "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"])
mrp = st.number_input("Product MRP", 0.0, 500.0, 150.0)

st.subheader("Store details")
store_id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
year = st.number_input("Store establishment year", 1980, 2025, 2005)
size = st.selectbox("Store size", ["Small", "Medium", "High"])
city = st.selectbox("City type", ["Tier 1", "Tier 2", "Tier 3"])
stype = st.selectbox("Store type", ["Supermarket Type1", "Supermarket Type2",
                                    "Departmental Store", "Food Mart"])

# Column names must match the training data exactly
input_df = pd.DataFrame([{
    "Product_Weight": weight, "Product_Sugar_Content": sugar,
    "Product_Allocated_Area": area, "Product_Type": ptype, "Product_MRP": mrp,
    "Store_Id": store_id, "Store_Establishment_Year": year, "Store_Size": size,
    "Store_Location_City_Type": city, "Store_Type": stype}])

if st.button("Predict"):
    st.write(input_df)
    st.success(f"Predicted sales: {model.predict(input_df)[0]:,.2f}")
