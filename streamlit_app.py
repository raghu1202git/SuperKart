import os
import requests
import streamlit as st

API_URL = os.getenv("SUPERKART_API_URL", "http://localhost:7860").rstrip("/")

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="wide")
st.title("🛒 SuperKart Quarterly Sales Forecast")
st.caption("Enter product and store attributes to estimate sales revenue.")

col1, col2 = st.columns(2)
with col1:
    product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
    sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")
    mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
    product_id_char = st.text_input("Product ID Prefix", value="FD", max_chars=2)
with col2:
    store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
    store_age = st.number_input("Store Age (Years)", min_value=0, value=16)
    category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

payload = {
    "Product_Weight": product_weight,
    "Product_Sugar_Content": sugar,
    "Product_Allocated_Area": area,
    "Product_MRP": mrp,
    "Store_Size": store_size,
    "Store_Location_City_Type": city_type,
    "Store_Type": store_type,
    "Product_Id_char": product_id_char,
    "Store_Age_Years": store_age,
    "Product_Type_Category": category,
}

if st.button("Predict Sales", type="primary"):
    try:
        response = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=30)
        if response.status_code == 200:
            value = response.json()["predicted_sales"]
            st.success(f"Estimated Product-Store Sales: ₹{value:,.2f}")
        else:
            st.error(f"API error {response.status_code}: {response.text}")
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {API_URL}: {exc}")

st.divider()
st.subheader("Batch Prediction")
uploaded = st.file_uploader("Upload batch CSV", type=["csv"])
if uploaded:
    if st.button("Run Batch Prediction"):
        try:
            response = requests.post(
                f"{API_URL}/v1/predictbatch",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                timeout=60,
            )
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except requests.RequestException as exc:
            st.error(f"Could not reach API at {API_URL}: {exc}")
