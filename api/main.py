from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional
import numpy as np
import shap
from pydantic import BaseModel


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning.",
    version="1.0.0"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

DATA_PATH = BASE_DIR / "data" / "telco_clean.csv"


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


print("\n==============================================")
print("MODEL DEBUG")
print("==============================================")

print("Model path:", MODEL_PATH)
print("Model type:", type(model))

if hasattr(model, "classes_"):
    print("Model classes:", model.classes_)


# =========================================================
# EXTRACT PREPROCESSOR + CLASSIFIER
# =========================================================

preprocessor = model[:-1]

classifier = model[-1]


print("\n==============================================")
print("PIPELINE DEBUG")
print("==============================================")

print("Preprocessor:", type(preprocessor))
print("Classifier:", type(classifier))


# =========================================================
# PREPARE SHAP BACKGROUND DATA
# =========================================================

print("\n==============================================")
print("SHAP SETUP")
print("==============================================")


try:

    # -----------------------------------------------------
    # Load cleaned dataset
    # -----------------------------------------------------

    df = pd.read_csv(DATA_PATH)


    # -----------------------------------------------------
    # Remove target column
    # -----------------------------------------------------

    X_background = df.drop(
        "Churn Label",
        axis=1
    )


    # -----------------------------------------------------
    # Use 500 samples as SHAP background
    # -----------------------------------------------------

    X_background = X_background.sample(
        n=min(500, len(X_background)),
        random_state=42
    )


    print(
        "SHAP background shape:",
        X_background.shape
    )


    # -----------------------------------------------------
    # Transform background data
    # -----------------------------------------------------

    background_encoded = preprocessor.transform(
        X_background
    )


    # -----------------------------------------------------
    # Convert sparse matrix to dense
    # -----------------------------------------------------

    if hasattr(
        background_encoded,
        "toarray"
    ):

        background_encoded = (
            background_encoded.toarray()
        )


    print(
        "Encoded background shape:",
        background_encoded.shape
    )


    # -----------------------------------------------------
    # Feature names
    # -----------------------------------------------------

    if hasattr(
        preprocessor,
        "get_feature_names_out"
    ):

        SHAP_FEATURE_NAMES = (
            preprocessor.get_feature_names_out()
        )

    else:

        SHAP_FEATURE_NAMES = [
            f"Feature_{i}"
            for i in range(
                background_encoded.shape[1]
            )
        ]


    print(
        "Number of SHAP features:",
        len(SHAP_FEATURE_NAMES)
    )


    # -----------------------------------------------------
    # Create ONE global SHAP explainer
    #
    # IMPORTANT:
    # We use the background dataset here.
    # We DO NOT use the customer itself as background.
    # -----------------------------------------------------

    SHAP_EXPLAINER = shap.LinearExplainer(
        classifier,
        background_encoded
    )


    print(
        "SHAP explainer created successfully."
    )


except Exception as e:

    print("\n========== SHAP SETUP ERROR ==========")

    print(str(e))

    SHAP_EXPLAINER = None

    SHAP_FEATURE_NAMES = []


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message":
            "Customer Churn Prediction API is running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "model_loaded":
            model is not None,

        "shap_loaded":
            SHAP_EXPLAINER is not None
    }


# =========================================================
# CUSTOMER DATA MODEL
# =========================================================

class CustomerData(BaseModel):

    # -----------------------------------------------------
    # Customer Information
    # -----------------------------------------------------

    Gender: str

    Age: int

    Under_30: str

    Senior_Citizen: str

    Married: str

    Dependents: str

    Number_of_Dependents: int


    # -----------------------------------------------------
    # Location
    # -----------------------------------------------------

    Country: str

    State: str

    City: str

    Zip_Code: int

    Latitude: float

    Longitude: float

    Population: int

    Quarter: str


    # -----------------------------------------------------
    # Customer History
    # -----------------------------------------------------

    Referred_a_Friend: str

    Number_of_Referrals: int

    Tenure_in_Months: int


    # -----------------------------------------------------
    # Offer
    # -----------------------------------------------------

    Offer: Optional[str] = None


    # -----------------------------------------------------
    # Phone
    # -----------------------------------------------------

    Phone_Service: str

    Avg_Monthly_Long_Distance_Charges: float

    Multiple_Lines: str


    # -----------------------------------------------------
    # Internet
    # -----------------------------------------------------

    Internet_Service: str

    Internet_Type: str

    Avg_Monthly_GB_Download: float


    # -----------------------------------------------------
    # Online Services
    # -----------------------------------------------------

    Online_Security: str

    Online_Backup: str

    Device_Protection_Plan: str

    Premium_Tech_Support: str


    # -----------------------------------------------------
    # Streaming
    # -----------------------------------------------------

    Streaming_TV: str

    Streaming_Movies: str

    Streaming_Music: str


    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

    Unlimited_Data: str


    # -----------------------------------------------------
    # Contract / Billing
    # -----------------------------------------------------

    Contract: str

    Paperless_Billing: str

    Payment_Method: str


    # -----------------------------------------------------
    # Charges
    # -----------------------------------------------------

    Monthly_Charge: float

    Total_Charges: float

    Total_Refunds: float

    Total_Extra_Data_Charges: float

    Total_Long_Distance_Charges: float


    # -----------------------------------------------------
    # Satisfaction
    # -----------------------------------------------------

    Satisfaction_Score: int


# =========================================================
# CREATE DATAFRAME
# =========================================================

def create_dataframe(
    customer: CustomerData
):

    data = pd.DataFrame([{

        # -------------------------------------------------
        # Customer Information
        # -------------------------------------------------

        "Gender":
            customer.Gender,

        "Age":
            customer.Age,

        "Under 30":
            customer.Under_30,

        "Senior Citizen":
            customer.Senior_Citizen,

        "Married":
            customer.Married,

        "Dependents":
            customer.Dependents,

        "Number of Dependents":
            customer.Number_of_Dependents,


        # -------------------------------------------------
        # Location
        # -------------------------------------------------

        "Country":
            customer.Country,

        "State":
            customer.State,

        "City":
            customer.City,

        "Zip Code":
            customer.Zip_Code,

        "Latitude":
            customer.Latitude,

        "Longitude":
            customer.Longitude,

        "Population":
            customer.Population,

        "Quarter":
            customer.Quarter,


        # -------------------------------------------------
        # Customer History
        # -------------------------------------------------

        "Referred a Friend":
            customer.Referred_a_Friend,

        "Number of Referrals":
            customer.Number_of_Referrals,

        "Tenure in Months":
            customer.Tenure_in_Months,


        # -------------------------------------------------
        # Offer
        # -------------------------------------------------

        "Offer":
            customer.Offer,


        # -------------------------------------------------
        # Phone
        # -------------------------------------------------

        "Phone Service":
            customer.Phone_Service,

        "Avg Monthly Long Distance Charges":
            customer.Avg_Monthly_Long_Distance_Charges,

        "Multiple Lines":
            customer.Multiple_Lines,


        # -------------------------------------------------
        # Internet
        # -------------------------------------------------

        "Internet Service":
            customer.Internet_Service,

        "Internet Type":
            customer.Internet_Type,

        "Avg Monthly GB Download":
            customer.Avg_Monthly_GB_Download,


        # -------------------------------------------------
        # Online Services
        # -------------------------------------------------

        "Online Security":
            customer.Online_Security,

        "Online Backup":
            customer.Online_Backup,

        "Device Protection Plan":
            customer.Device_Protection_Plan,

        "Premium Tech Support":
            customer.Premium_Tech_Support,


        # -------------------------------------------------
        # Streaming
        # -------------------------------------------------

        "Streaming TV":
            customer.Streaming_TV,

        "Streaming Movies":
            customer.Streaming_Movies,

        "Streaming Music":
            customer.Streaming_Music,


        # -------------------------------------------------
        # Data
        # -------------------------------------------------

        "Unlimited Data":
            customer.Unlimited_Data,


        # -------------------------------------------------
        # Contract / Billing
        # -------------------------------------------------

        "Contract":
            customer.Contract,

        "Paperless Billing":
            customer.Paperless_Billing,

        "Payment Method":
            customer.Payment_Method,


        # -------------------------------------------------
        # Charges
        # -------------------------------------------------

        "Monthly Charge":
            customer.Monthly_Charge,

        "Total Charges":
            customer.Total_Charges,

        "Total Refunds":
            customer.Total_Refunds,

        "Total Extra Data Charges":
            customer.Total_Extra_Data_Charges,

        "Total Long Distance Charges":
            customer.Total_Long_Distance_Charges,


        # -------------------------------------------------
        # Satisfaction
        # -------------------------------------------------

        "Satisfaction Score":
            customer.Satisfaction_Score

    }])


    # -----------------------------------------------------
    # Convert None to NaN
    # -----------------------------------------------------

    data = data.replace(
        {None: np.nan}
    )


    return data


# =========================================================
# CLEAN SHAP FEATURE NAME
# =========================================================

def clean_feature_name(
    feature_name
):

    feature_name = str(
        feature_name
    )


    # -----------------------------------------------------
    # Remove numerical transformer prefix
    # -----------------------------------------------------

    if feature_name.startswith(
        "num__"
    ):

        feature_name = feature_name[
            len("num__"):
        ]


    # -----------------------------------------------------
    # Remove categorical transformer prefix
    # -----------------------------------------------------

    elif feature_name.startswith(
        "cat__"
    ):

        feature_name = feature_name[
            len("cat__"):
        ]


    return feature_name


# =========================================================
# SHAP EXPLANATION
# =========================================================

def get_shap_explanation(
    data
):

    try:

        # -------------------------------------------------
        # Check SHAP explainer
        # -------------------------------------------------

        if SHAP_EXPLAINER is None:

            print(
                "SHAP explainer is not available."
            )

            return []


        # -------------------------------------------------
        # Transform customer
        # -------------------------------------------------

        data_encoded = (
            preprocessor.transform(data)
        )


        # -------------------------------------------------
        # Convert sparse to dense
        # -------------------------------------------------

        if hasattr(
            data_encoded,
            "toarray"
        ):

            data_encoded = (
                data_encoded.toarray()
            )


        # -------------------------------------------------
        # Calculate SHAP values
        # -------------------------------------------------

        shap_values = SHAP_EXPLAINER(
            data_encoded
        )


        # -------------------------------------------------
        # Extract values
        # -------------------------------------------------

        values = shap_values.values


        # -------------------------------------------------
        # Make sure values are 1D
        # -------------------------------------------------

        if len(values.shape) == 2:

            values = values[0]


        # -------------------------------------------------
        # Create explanation dataframe
        # -------------------------------------------------

        explanation = pd.DataFrame({

            "feature":
                SHAP_FEATURE_NAMES,

            "shap_value":
                values

        })


        # -------------------------------------------------
        # Absolute impact
        # -------------------------------------------------

        explanation[
            "absolute_impact"
        ] = explanation[
            "shap_value"
        ].abs()


        # -------------------------------------------------
        # Clean feature names
        # -------------------------------------------------

        explanation[
            "feature"
        ] = explanation[
            "feature"
        ].apply(
            clean_feature_name
        )


        # -------------------------------------------------
        # Sort by absolute importance
        # -------------------------------------------------

        explanation = explanation.sort_values(
            "absolute_impact",
            ascending=False
        )


        # -------------------------------------------------
        # Top 10 features
        # -------------------------------------------------

        top_features = explanation.head(
            10
        )


        # -------------------------------------------------
        # API response
        # -------------------------------------------------

        results = []


        for _, row in top_features.iterrows():

            shap_value = float(
                row["shap_value"]
            )


            # ---------------------------------------------
            # Determine impact
            # ---------------------------------------------

            if shap_value > 0:

                direction = (
                    "increases_churn"
                )

            elif shap_value < 0:

                direction = (
                    "decreases_churn"
                )

            else:

                direction = (
                    "no_effect"
                )


            # ---------------------------------------------
            # Add result
            # ---------------------------------------------

            results.append({

                "feature":
                    str(row["feature"]),

                "shap_value":
                    round(
                        shap_value,
                        4
                    ),

                "impact":
                    direction,

                "absolute_impact":
                    round(
                        float(
                            row[
                                "absolute_impact"
                            ]
                        ),
                        4
                    )

            })


        return results


    except Exception as e:

        print(
            "\n========== SHAP ERROR =========="
        )

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        return []


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict_churn(
    customer: CustomerData
):

    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    data = create_dataframe(
        customer
    )


    # =====================================================
    # DEBUG DATA
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        "API DATA"
    )

    print(
        "=============================================="
    )

    print(
        data.to_string()
    )

    print(
        "\nAPI SHAPE:",
        data.shape
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        "API PREDICTION"
    )

    print(
        "=============================================="
    )


    prediction = model.predict(
        data
    )[0]


    probabilities = model.predict_proba(
        data
    )[0]


    print(
        "Prediction:",
        prediction
    )


    print(
        "Probabilities:",
        probabilities
    )


    # =====================================================
    # CLASS DEBUG
    # =====================================================

    if hasattr(
        model,
        "classes_"
    ):

        print(
            "\nClass mapping:"
        )


        for cls, prob in zip(
            model.classes_,
            probabilities
        ):

            print(
                f"Class {cls}: "
                f"{prob * 100:.2f}%"
            )


    # =====================================================
    # CHURN PROBABILITY
    # =====================================================

    # Your model classes are [0, 1]
    # Therefore index 1 = Churn

    probability = probabilities[1]


    print(
        "\nSelected churn probability:",
        probability * 100
    )


    # =====================================================
    # SHAP EXPLANATION
    # =====================================================

    print(
        "\n=============================================="
    )

    print(
        "SHAP EXPLANATION"
    )

    print(
        "=============================================="
    )


    shap_explanation = (
        get_shap_explanation(data)
    )


    # =====================================================
    # PRINT SHAP RESULTS
    # =====================================================

    for item in shap_explanation:

        print(
            item["feature"],
            "=>",
            item["shap_value"],
            item["impact"]
        )


    # =====================================================
    # RETURN RESPONSE
    # =====================================================

    return {

        "prediction":
            "Churn"
            if prediction == 1
            else "No Churn",


        "churn_probability":
            round(
                float(probability) * 100,
                2
            ),


        "shap_explanation":
            shap_explanation

    }