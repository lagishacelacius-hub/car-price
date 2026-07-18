import streamlit as st
import pandas as pd
import joblib
import time

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Car Price Advisor",
    page_icon="🚗",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("model/car_price_model.pkl")

# -------------------------------------------------
# BRAND & MODEL DATA
# -------------------------------------------------
brand_models = {
    "Tesla": ["Model 3", "Model S", "Model X", "Model Y"],
    "BMW": ["3 Series", "5 Series", "X3", "X5"],
    "Audi": ["A3", "A4", "Q5", "Q7"],
    "Ford": ["Explorer", "Fiesta", "Focus", "Mustang"],
    "Honda": ["Accord", "CR-V", "Civic", "Fit"],
    "Mercedes": ["C-Class", "E-Class", "GLA", "GLC"],
    "Toyota": ["Camry", "Corolla", "Prius", "RAV4"]
}

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:

    st.header("🤖 Model Information")

    st.success("Machine Learning Model")

    st.write("""
**Algorithm**
- Random Forest Regression

**Dataset**
- 2500 Car Records

**Features**
- Brand
- Model
- Year
- Engine Size
- Fuel Type
- Transmission
- Mileage
- Condition

**Purpose**
Estimate the market price of a used car.
""")

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🚗 Smart Car Price Advisor")

st.markdown(
"""
Predict the estimated market value of a car using
**Machine Learning** and receive smart recommendations
for buying or selling.
"""
)

st.divider()

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------

purpose = st.radio(
    "What are you planning to do?",
    ["Buy a Car", "Sell a Car"],
    horizontal=True
)

col1, col2 = st.columns(2)

with col1:

    brand = st.selectbox(
        "🚗 Brand",
        list(brand_models.keys())
    )

    model_name = st.selectbox(
        "🚘 Model",
        brand_models[brand]
    )

    year = st.number_input(
        "Manufacturing Year",
        1990,
        2026,
        2020
    )

    engine_size = st.number_input(
        "Engine Size (L)",
        0.5,
        10.0,
        2.0
    )

with col2:

    fuel = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "Electric", "Hybrid"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    mileage = st.number_input(
        "Mileage (km)",
        min_value=0,
        value=30000
    )

    condition = st.selectbox(
        "Condition",
        ["New", "Like New", "Used"]
    )

expected_price = st.number_input(
    "Expected Price ($) (Optional)",
    min_value=0,
    value=30000
)

st.divider()

# -------------------------------------------------
# PREDICT
# -------------------------------------------------

if st.button("🚀 Predict Car Price", use_container_width=True):

    input_data = pd.DataFrame({

        "Brand": [brand],
        "Year": [year],
        "Engine Size": [engine_size],
        "Fuel Type": [fuel],
        "Transmission": [transmission],
        "Mileage": [mileage],
        "Condition": [condition],
        "Model": [model_name]

    })

    with st.spinner("🤖 AI is analysing your car..."):
        time.sleep(2)
        prediction = model.predict(input_data)[0]

    lower_price = prediction * 0.90
    upper_price = prediction * 1.10

    # ---------------------------------------------
    # Category
    # ---------------------------------------------

    if prediction < 20000:
        category = "🚙 Budget Car"

    elif prediction < 50000:
        category = "🚘 Mid-range Car"

    else:
        category = "🏎️ Premium Car"

    # ---------------------------------------------
    # Health Score
    # ---------------------------------------------

    score = 100

    if mileage > 100000:
        score -= 25
    elif mileage > 50000:
        score -= 10

    if year < 2015:
        score -= 15

    if condition == "Used":
        score -= 15

    elif condition == "Like New":
        score -= 5

    score = max(score, 0)

    # ---------------------------------------------
    # Good Deal Checker
    # ---------------------------------------------

    if expected_price < lower_price:

        verdict = "🟢 Excellent Deal"

    elif lower_price <= expected_price <= upper_price:

        verdict = "🟡 Fair Price"

    else:

        verdict = "🔴 Overpriced"

    # ---------------------------------------------
    # Advice
    # ---------------------------------------------

    if purpose == "Buy a Car":

        advice = """
✔ Compare prices with similar models.

✔ Check accident history.

✔ Verify service records.

✔ Take a test drive before buying.
"""

    else:

        advice = """
✔ Keep maintenance records.

✔ Clean the vehicle before selling.

✔ Highlight recent servicing.

✔ Take quality photos for resale.
"""

    # ---------------------------------------------
    # Results
    # ---------------------------------------------

    st.divider()

    st.success(f"💰 Estimated Price : ${prediction:,.2f}")

    st.info(
        f"📊 Market Price Range : ${lower_price:,.2f}  -  ${upper_price:,.2f}"
    )

    st.write(f"### {category}")

    st.metric(
        "🚗 Car Health Score",
        f"{score}/100"
    )

    st.metric(
        "💵 Deal Evaluation",
        verdict
    )

    st.warning(advice)

    st.write("### 📋 Selected Vehicle")

    st.write(f"**Brand:** {brand}")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Fuel:** {fuel}")
    st.write(f"**Transmission:** {transmission}")
    st.write(f"**Condition:** {condition}")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Developed by Lagisha Celacius | "
    "B.Tech Artificial Intelligence & Data Science | "
    "Machine Learning Project"
)