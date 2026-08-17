import streamlit as st
import requests
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "reports"

CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = REPORTS_DIR / "roc_curve.png"


# =========================================================
# PAGE TITLE
# =========================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn using "
    "the trained machine learning model."
)

st.divider()


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.header("📈 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Accuracy",
        "97%"
    )

with col2:
    st.metric(
        "Precision",
        "97%"
    )

with col3:
    st.metric(
        "Recall",
        "90%"
    )

with col4:
    st.metric(
        "F1 Score",
        "93%"
    )

with col5:
    st.metric(
        "ROC-AUC",
        "0.9932"
    )

st.caption(
    "Performance evaluated on the test dataset."
)

st.divider()


# =========================================================
# MODEL EVALUATION
# =========================================================

st.header("📊 Model Evaluation")

col1, col2 = st.columns(2)


# =========================================================
# CONFUSION MATRIX
# =========================================================

with col1:

    st.subheader("Confusion Matrix")

    if CONFUSION_MATRIX_PATH.exists():

        st.image(
            str(CONFUSION_MATRIX_PATH),
            width="stretch"
        )

    else:

        st.error(
            "Confusion matrix image was not found."
        )

        st.caption(
            str(CONFUSION_MATRIX_PATH)
        )


# =========================================================
# ROC CURVE
# =========================================================

with col2:

    st.subheader("ROC Curve")

    if ROC_CURVE_PATH.exists():

        st.image(
            str(ROC_CURVE_PATH),
            width="stretch"
        )

    else:

        st.error(
            "ROC curve image was not found."
        )

        st.caption(
            str(ROC_CURVE_PATH)
        )


st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


# =========================================================
# CUSTOMER INFORMATION - COLUMN 1
# =========================================================

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        index=0
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=78
    )

    under_30 = st.selectbox(
        "Under 30",
        ["Yes", "No"],
        index=1
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["Yes", "No"],
        index=0
    )


# =========================================================
# CUSTOMER INFORMATION - COLUMN 2
# =========================================================

with col2:

    married = st.selectbox(
        "Married",
        ["Yes", "No"],
        index=1
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"],
        index=1
    )

    number_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        value=0
    )

    satisfaction_score = st.number_input(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )


# =========================================================
# CUSTOMER INFORMATION - COLUMN 3
# =========================================================

with col3:

    tenure = st.number_input(
        "Tenure in Months",
        min_value=0,
        value=1
    )

    monthly_charge = st.number_input(
        "Monthly Charge",
        min_value=0.0,
        value=39.65,
        step=0.01
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=39.65,
        step=0.01
    )


st.divider()


# =========================================================
# SERVICE INFORMATION
# =========================================================

st.header("📱 Service Information")

col1, col2, col3 = st.columns(3)


# =========================================================
# SERVICE COLUMN 1
# =========================================================

with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
        index=1
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No"],
        index=1
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["Yes", "No"],
        index=0
    )


# =========================================================
# SERVICE COLUMN 2
# =========================================================

with col2:

    internet_type = st.selectbox(
        "Internet Type",
        ["DSL", "Fiber Optic", "Cable", "None"],
        index=0
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No"],
        index=1
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No"],
        index=1
    )


# =========================================================
# SERVICE COLUMN 3
# =========================================================

with col3:

    device_protection = st.selectbox(
        "Device Protection Plan",
        ["Yes", "No"],
        index=0
    )

    premium_support = st.selectbox(
        "Premium Tech Support",
        ["Yes", "No"],
        index=1
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-Month", "One year", "Two year"],
        index=0
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔮 Predict Churn",
    type="primary",
    width="stretch"
):

    # =====================================================
    # CUSTOMER DATA
    # =====================================================

    customer_data = {

        # -------------------------------------------------
        # CUSTOMER INFORMATION
        # -------------------------------------------------

        "Gender": gender,
        "Age": age,
        "Under_30": under_30,
        "Senior_Citizen": senior_citizen,
        "Married": married,
        "Dependents": dependents,
        "Number_of_Dependents": number_of_dependents,

        # -------------------------------------------------
        # LOCATION
        # -------------------------------------------------

        "Country": "United States",
        "State": "California",
        "City": "Los Angeles",
        "Zip_Code": 90022,
        "Latitude": 34.02381,
        "Longitude": -118.156582,
        "Population": 68701,
        "Quarter": "Q3",

        # -------------------------------------------------
        # CUSTOMER HISTORY
        # -------------------------------------------------

        "Referred_a_Friend": "No",
        "Number_of_Referrals": 0,
        "Tenure_in_Months": tenure,

        # -------------------------------------------------
        # OFFER
        # -------------------------------------------------

        "Offer": None,

        # -------------------------------------------------
        # PHONE
        # -------------------------------------------------

        "Phone_Service": phone_service,
        "Avg_Monthly_Long_Distance_Charges": 0.0,
        "Multiple_Lines": multiple_lines,

        # -------------------------------------------------
        # INTERNET
        # -------------------------------------------------

        "Internet_Service": internet_service,
        "Internet_Type": internet_type,
        "Avg_Monthly_GB_Download": 8.0,

        # -------------------------------------------------
        # ONLINE SERVICES
        # -------------------------------------------------

        "Online_Security": online_security,
        "Online_Backup": online_backup,
        "Device_Protection_Plan": device_protection,
        "Premium_Tech_Support": premium_support,

        # -------------------------------------------------
        # STREAMING
        # -------------------------------------------------

        "Streaming_TV": "No",
        "Streaming_Movies": "Yes",
        "Streaming_Music": "No",

        # -------------------------------------------------
        # DATA
        # -------------------------------------------------

        "Unlimited_Data": "No",

        # -------------------------------------------------
        # CONTRACT / BILLING
        # -------------------------------------------------

        "Contract": contract,
        "Paperless_Billing": "Yes",
        "Payment_Method": "Bank Withdrawal",

        # -------------------------------------------------
        # CHARGES
        # -------------------------------------------------

        "Monthly_Charge": monthly_charge,
        "Total_Charges": total_charges,
        "Total_Refunds": 0.0,
        "Total_Extra_Data_Charges": 20.0,
        "Total_Long_Distance_Charges": 0.0,

        # -------------------------------------------------
        # SATISFACTION
        # -------------------------------------------------

        "Satisfaction_Score": satisfaction_score
    }


    # =====================================================
    # SHOW DATA SENT TO API
    # =====================================================

    with st.expander(
        "🔍 View Customer Data Sent to API"
    ):

        st.json(customer_data)


    # =====================================================
    # CALL FASTAPI
    # =====================================================

    try:

        with st.spinner(
            "Analyzing customer data..."
        ):

            response = requests.post(
                "https://customer-churn-prediction-tn0a.onrender.com/predict",
                json=customer_data,
                timeout=60
            )


        # =================================================
        # SUCCESS
        # =================================================

        if response.status_code == 200:

            result = response.json()

            prediction = result.get(
                "prediction",
                "Unknown"
            )

            probability = float(
                result.get(
                    "churn_probability",
                    0
                )
            )

            shap_explanation = result.get(
                "shap_explanation",
                []
            )


            # =================================================
            # PREDICTION RESULT
            # =================================================

            st.divider()

            st.header(
                "📊 Prediction Result"
            )


            # =================================================
            # MAIN PREDICTION
            # =================================================

            if prediction == "Churn":

                st.error(
                    f"🔴 Prediction: {prediction}"
                )

            elif prediction == "No Churn":

                st.success(
                    f"🟢 Prediction: {prediction}"
                )

            else:

                st.warning(
                    f"⚪ Prediction: {prediction}"
                )


            # =================================================
            # PROBABILITY
            # =================================================

            st.metric(
                "🎯 Churn Probability",
                f"{probability:.2f}%"
            )


            # =================================================
            # PROGRESS BAR
            # =================================================

            st.progress(
                min(
                    max(
                        probability / 100,
                        0.0
                    ),
                    1.0
                )
            )


            # =================================================
            # RISK LEVEL
            # =================================================

            if probability >= 70:

                risk_level = "Very High"

                st.error(
                    "⚠️ Very High Churn Risk"
                )

            elif probability >= 50:

                risk_level = "High"

                st.warning(
                    "⚠️ High Churn Risk"
                )

            elif probability >= 30:

                risk_level = "Moderate"

                st.warning(
                    "🟡 Moderate Churn Risk"
                )

            else:

                risk_level = "Low"

                st.success(
                    "🟢 Low Churn Risk"
                )


            # =================================================
            # CUSTOMER RISK SUMMARY
            # =================================================

            st.divider()

            st.subheader(
                "📋 Customer Risk Summary"
            )

            summary_col1, summary_col2, summary_col3 = (
                st.columns(3)
            )


            with summary_col1:

                st.metric(
                    "Prediction",
                    prediction
                )


            with summary_col2:

                st.metric(
                    "Churn Probability",
                    f"{probability:.2f}%"
                )


            with summary_col3:

                st.metric(
                    "Risk Level",
                    risk_level
                )


            # =================================================
            # SHAP EXPLANATION
            # =================================================

            st.divider()

            st.header(
                "💡 Why did the model make this prediction?"
            )


            if shap_explanation:

                st.write(
                    "The features below show which factors "
                    "had the greatest influence on this "
                    "customer's prediction."
                )


                # =============================================
                # CLEAN SHAP DATA
                # =============================================

                clean_shap = []

                for item in shap_explanation:

                    try:

                        feature = str(
                            item.get(
                                "feature",
                                "Unknown Feature"
                            )
                        )

                        shap_value = float(
                            item.get(
                                "shap_value",
                                0
                            )
                        )

                        absolute_impact = abs(
                            shap_value
                        )

                        impact = item.get(
                            "impact",
                            ""
                        )

                        clean_shap.append(
                            {
                                "feature": feature,
                                "shap_value": shap_value,
                                "absolute_impact": absolute_impact,
                                "impact": impact
                            }
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        continue


                # =============================================
                # SORT BY IMPORTANCE
                # =============================================

                clean_shap = sorted(
                    clean_shap,
                    key=lambda x: x["absolute_impact"],
                    reverse=True
                )


                # =============================================
                # TOP 10 FEATURES
                # =============================================

                top_shap = clean_shap[:10]


                # =============================================
                # SHAP SUMMARY TABLE
                # =============================================

                st.subheader(
                    "🔎 Top Factors"
                )

                shap_table = []

                for item in top_shap:

                    if item["shap_value"] > 0:

                        direction = (
                            "🔴 Increases churn risk"
                        )

                    elif item["shap_value"] < 0:

                        direction = (
                            "🟢 Decreases churn risk"
                        )

                    else:

                        direction = (
                            "⚪ No measurable effect"
                        )

                    shap_table.append(
                        {
                            "Feature": item["feature"],
                            "SHAP Value": round(
                                item["shap_value"],
                                4
                            ),
                            "Impact": direction
                        }
                    )


                if shap_table:

                    st.dataframe(
                        shap_table,
                        width="stretch",
                        hide_index=True
                    )


                # =============================================
                # SHAP VISUALIZATION
                # =============================================

                st.subheader(
                    "📊 Feature Impact"
                )

                for item in top_shap:

                    feature = item["feature"]

                    shap_value = item["shap_value"]

                    if shap_value > 0:

                        st.error(
                            f"🔴 **{feature}**  \n"
                            f"Increases churn risk — "
                            f"SHAP: `{shap_value:.4f}`"
                        )

                    elif shap_value < 0:

                        st.success(
                            f"🟢 **{feature}**  \n"
                            f"Decreases churn risk — "
                            f"SHAP: `{shap_value:.4f}`"
                        )

                    else:

                        st.info(
                            f"⚪ **{feature}**  \n"
                            f"No measurable effect — "
                            f"SHAP: `0.0000`"
                        )


            else:

                st.info(
                    "ℹ️ No SHAP explanation was returned "
                    "by the API."
                )


        # =================================================
        # API ERROR
        # =================================================

        else:

            st.error(
                f"❌ API Error: {response.status_code}"
            )

            st.code(
                response.text
            )


    # =====================================================
    # CONNECTION ERROR
    # =====================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FastAPI."
        )

        st.info(
            "Make sure the FastAPI backend is live on Render and "
            "the /predict endpoint is available."
        )


    # =====================================================
    # TIMEOUT ERROR
    # =====================================================

    except requests.exceptions.Timeout:

        st.error(
            "❌ FastAPI took too long to respond."
        )


    # =====================================================
    # INVALID RESPONSE
    # =====================================================

    except ValueError as e:

        st.error(
            "❌ The API returned an invalid response."
        )

        st.code(
            str(e)
        )


    # =====================================================
    # OTHER REQUEST ERRORS
    # =====================================================

    except requests.exceptions.RequestException as e:

        st.error(
            f"❌ Request failed: {e}"
        )


    # =====================================================
    # UNEXPECTED ERROR
    # =====================================================

    except Exception as e:

        st.error(
            f"❌ Unexpected error: {e}"
        )