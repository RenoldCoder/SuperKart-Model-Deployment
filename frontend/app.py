import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="SuperKart Sales Forecast", layout="wide")
st.title("SuperKart Quarterly Sales Forecast")
API_URL = st.sidebar.text_input("Backend API URL", "https://<your-backend-space>.hf.space/v1/predict")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
        Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.05, step=0.01)
        Product_MRP = st.number_input("Product MRP", min_value=0.0, value=141.62)
        Product_Id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])
    with c2:
        Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        Store_Location_City_Type = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        Store_Type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
        Store_Age_Years = st.number_input("Store Age Years", min_value=0, max_value=50, value=16)
        Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
    payload = {
        "Product_Weight": Product_Weight, "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area, "Product_MRP": Product_MRP,
        "Store_Size": Store_Size, "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type, "Product_Id_char": Product_Id_char,
        "Store_Age_Years": Store_Age_Years, "Product_Type_Category": Product_Type_Category,
    }
    if st.button("Predict Sales", type="primary"):
        response = requests.post(API_URL, json=payload, timeout=20)
        if response.ok:
            pred = response.json()["predictions"][0]
            st.success(f"Predicted Product Store Sales Total: {pred:,.2f}")
        else:
            st.error(response.text)
with tab2:
    uploaded = st.file_uploader("Upload a CSV with the required columns", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Run Batch Prediction"):
            response = requests.post(API_URL, json=df.to_dict(orient="records"), timeout=60)
            if response.ok:
                out = df.copy(); out["Predicted_Sales"] = response.json()["predictions"]
                st.dataframe(out.head())
                st.download_button("Download Predictions", out.to_csv(index=False), "superkart_batch_predictions.csv")
            else:
                st.error(response.text)