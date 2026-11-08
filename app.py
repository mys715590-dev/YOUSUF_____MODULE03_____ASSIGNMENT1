from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px

# ---------------- PATHS ----------------

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DATA = ROOT / "data" / "Churn_Modelling.csv"

FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember",
    "Geography", "Gender"
]

MODEL_FILES = {
    "Balanced ANN": "salary_ann_balanced.keras",
    "Deep ANN": "salary_ann_deep.keras",
    "Wide ANN": "salary_ann_wide.keras"
}

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Customer Salary Intelligence Pro",
    page_icon="📊",
    layout="wide"
)

# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #061326,
        #0a1e38 55%,
        #102b50
    );
    color: #edf5ff;
}
section[data-testid="stSidebar"] {
    background: #07172b;
}
h1, h2, h3 {
    color: #edf5ff;
}
div[data-testid="stMetric"] {
    background: #102b4b;
    padding: 16px;
    border: 1px solid #1765b5;
    border-radius: 13px;
}
.stButton > button {
    background: #087cf0;
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD SAVED ASSETS ----------------

@st.cache_resource
def load_assets():

    preprocessor = joblib.load(
        ART / "preprocessor.joblib"
    )

    feature_order = joblib.load(
        ART / "feature_order.joblib"
    )

    models = {}

    for label, filename in MODEL_FILES.items():

        path = ART / filename

        if path.exists():

            models[label] = tf.keras.models.load_model(
                path
            )

    return preprocessor, feature_order, models


# ---------------- SAMPLE PROFILES ----------------

def sample_profile(profile):

    profiles = {

        "Balanced customer": [
            650, 35, 5, 50000, 2, 1, 1,
            "France", "Female"
        ],

        "Young customer": [
            700, 25, 2, 15000, 1, 1, 1,
            "Spain", "Male"
        ],

        "Long-term customer": [
            720, 48, 10, 110000, 2, 1, 1,
            "Germany", "Female"
        ],

        "Low-balance customer": [
            600, 31, 4, 0, 1, 0, 0,
            "France", "Male"
        ]
    }

    return dict(
        zip(FEATURES, profiles[profile])
    )


# ---------------- PREDICTION FUNCTION ----------------

def predict_salary(row, model, preprocessor, feature_order):

    frame = pd.DataFrame([row])

    # Keep the exact training feature order
    frame = frame[feature_order]

    transformed = preprocessor.transform(frame)

    prediction = model.predict(
        transformed,
        verbose=0
    ).reshape(-1)[0]

    return float(prediction)


# ---------------- HEADER ----------------

st.title("📊 Customer Salary Intelligence Pro")

st.caption(
    "Advanced ANN Regression Laboratory | "
    "Customer Profile Analysis | Scenario Comparison"
)

st.info(
    """
    **DATA INTEGRITY NOTICE**

    This application is designed for academic and research purposes.
    EstimatedSalary represents an estimated dataset value, not verified
    customer income. Predictions must not be used for banking, lending,
    hiring, or other consequential decisions.
    """,
    icon="🔒"
)


# ---------------- CHECK ARTIFACTS ----------------

required_files = [
    ART / "preprocessor.joblib",
    ART / "feature_order.joblib"
]

if any(not path.exists() for path in required_files):

    st.error(
        "Preprocessing artifacts are missing. "
        "Run `python train_model.py` from the project folder first."
    )

    st.stop()


try:

    preprocessor, feature_order, models = load_assets()

except Exception as error:

    st.error(f"Could not load saved artifacts: {error}")
    st.stop()


if not models:

    st.error(
        "No trained ANN model found. "
        "Run the training script first."
    )

    st.stop()


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("Control Center")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Salary Predictor",
            "Compare Customer Scenarios",
            "Bulk Prediction",
            "Dataset Explorer",
            "Model Laboratory",
            "Research Notes"
        ]
    )

    chosen = st.selectbox(
        "Active ANN model",
        list(models.keys())
    )

    st.caption(
        "Select a trained architecture. "
        "All models use the same saved preprocessing pipeline."
    )


model = models[chosen]


# ---------------- METRICS REPORT ----------------

metrics_path = ART / "metrics.json"

if metrics_path.exists():

    report = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )

else:

    report = {}


# =====================================================
# 1. OVERVIEW
# =====================================================

if page == "Overview":

    st.subheader("Executive Overview")

    if report:

        metric = report.get(
            "models", {}
        ).get(chosen, {})

        a, b, c, d = st.columns(4)

        a.metric(
            "ANN Test MAE",
            f"{metric.get('MAE', 0):,.2f}"
        )

        b.metric(
            "ANN Test RMSE",
            f"{metric.get('RMSE', 0):,.2f}"
        )

        c.metric(
            "ANN Test R²",
            f"{metric.get('R2', 0):.3f}"
        )

        d.metric(
            "Architectures",
            str(len(models))
        )

    else:

        st.info(
            "Evaluation report will appear after model training."
        )

    st.markdown("### Application Modules")

    cards = [
        (
            "01",
            "Salary Predictor",
            "Single customer estimate with editable profile inputs"
        ),
        (
            "02",
            "Scenario Comparison",
            "Compare multiple customer profiles"
        ),
        (
            "03",
            "Bulk Prediction",
            "Upload CSV and export predictions"
        ),
        (
            "04",
            "Dataset Explorer",
            "Explore distributions and correlations"
        ),
        (
            "05",
            "Model Laboratory",
            "Compare ANN architecture performance"
        ),
        (
            "06",
            "Research Notes",
            "Review methodology and limitations"
        )
    ]

    columns = st.columns(3)

    for index, (number, title, description) in enumerate(cards):

        with columns[index % 3]:

            st.markdown(
                f"**{number} · {title}**"
            )

            st.caption(description)

    st.info(
        "Select a module from the sidebar to explore the application."
    )


# =====================================================
# 2. INDIVIDUAL SALARY PREDICTOR
# =====================================================

elif page == "Salary Predictor":

    st.subheader("Individual Customer Salary Prediction")

    left, right = st.columns([1.2, 1])

    with left:

        profile = st.selectbox(
            "Start from a sample profile",
            [
                "Custom profile",
                "Balanced customer",
                "Young customer",
                "Long-term customer",
                "Low-balance customer"
            ]
        )

        if profile == "Custom profile":

            defaults = sample_profile(
                "Balanced customer"
            )

        else:

            defaults = sample_profile(profile)

        with st.form("single_prediction_form"):

            c1, c2, c3 = st.columns(3)

            with c1:

                credit = st.slider(
                    "Credit Score",
                    300, 900,
                    int(defaults["CreditScore"])
                )

                age = st.slider(
                    "Age",
                    18, 100,
                    int(defaults["Age"])
                )

                tenure = st.slider(
                    "Tenure",
                    0, 10,
                    int(defaults["Tenure"])
                )

            with c2:

                balance = st.number_input(
                    "Balance",
                    min_value=0.0,
                    max_value=250000.0,
                    value=float(defaults["Balance"]),
                    step=1000.0
                )

                products = st.selectbox(
                    "Number of Products",
                    [1, 2, 3, 4],
                    index=int(defaults["NumOfProducts"]) - 1
                )

                geography = st.selectbox(
                    "Geography",
                    ["France", "Germany", "Spain"],
                    index=[
                        "France",
                        "Germany",
                        "Spain"
                    ].index(defaults["Geography"])
                )

            with c3:

                gender = st.selectbox(
                    "Gender",
                    ["Female", "Male"],
                    index=[
                        "Female",
                        "Male"
                    ].index(defaults["Gender"])
                )

                card = st.toggle(
                    "Has Credit Card",
                    value=bool(defaults["HasCrCard"])
                )

                active = st.toggle(
                    "Active Member",
                    value=bool(defaults["IsActiveMember"])
                )

            submit = st.form_submit_button(
                "Generate Salary Estimate",
                use_container_width=True
            )

    with right:

        st.markdown("### Prediction Output")

        if submit:

            row = {
                "CreditScore": credit,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": products,
                "HasCrCard": int(card),
                "IsActiveMember": int(active),
                "Geography": geography,
                "Gender": gender
            }

            value = predict_salary(
                row,
                model,
                preprocessor,
                feature_order
            )

            st.metric(
                "Predicted EstimatedSalary",
                f"{value:,.2f}"
            )

            st.caption(
                f"Selected model: {chosen}"
            )

            st.success("Prediction completed.")

        else:

            st.info(
                "Choose a profile, adjust the inputs "
                "and generate an estimate."
            )

        st.caption(
            "This is a model output, not a verified salary."
        )


# =====================================================
# 3. CUSTOMER SCENARIO COMPARISON
# =====================================================

elif page == "Compare Customer Scenarios":

    st.subheader("What-if Scenario Comparison")

    st.write(
        "Compare preset customer profiles using the selected ANN."
    )

    names = [
        "Balanced customer",
        "Young customer",
        "Long-term customer",
        "Low-balance customer"
    ]

    selected = st.multiselect(
        "Select customer scenarios",
        names,
        default=names[:3]
    )

    if selected:

        rows = []

        for name in selected:

            row = sample_profile(name)

            prediction = predict_salary(
                row,
                model,
                preprocessor,
                feature_order
            )

            rows.append({
                "Scenario": name,
                "Predicted salary": prediction,
                **row
            })

        result = pd.DataFrame(rows)

        st.dataframe(
            result[
                ["Scenario", "Predicted salary"] + FEATURES
            ],
            use_container_width=True
        )

        fig = px.bar(
            result,
            x="Scenario",
            y="Predicted salary",
            title="Predicted Salary by Scenario"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.download_button(
            "Download Scenario Comparison",
            result.to_csv(index=False).encode(),
            "scenario_comparison.csv",
            "text/csv"
        )


# =====================================================
# 4. BULK PREDICTION
# =====================================================

elif page == "Bulk Prediction":

    st.subheader("Bulk Customer Prediction")

    st.write(
        "Upload a CSV containing the nine model input columns. "
        "The target column is not required."
    )

    st.code(", ".join(FEATURES))

    uploaded = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded:

        data = pd.read_csv(uploaded)

        missing_cols = [
            col for col in FEATURES
            if col not in data.columns
        ]

        if missing_cols:

            st.error(
                f"Missing columns: {missing_cols}"
            )

        else:

            clean = data[FEATURES].copy()

            values = model.predict(
                preprocessor.transform(clean),
                verbose=0
            ).reshape(-1)

            output = data.copy()

            output["PredictedEstimatedSalary"] = values

            st.metric(
                "Customers Processed",
                len(output)
            )

            st.dataframe(
                output.head(200),
                use_container_width=True
            )

            st.download_button(
                "Download All Predictions",
                output.to_csv(index=False).encode(),
                "customer_salary_predictions.csv",
                "text/csv"
            )


# =====================================================
# 5. DATASET EXPLORER
# =====================================================

elif page == "Dataset Explorer":

    st.subheader("Dataset Intelligence")

    if not DATA.exists():

        st.info(
            "Add Churn_Modelling.csv to the data folder "
            "to enable dataset exploration."
        )

    else:

        df = pd.read_csv(DATA)

        x1, x2, x3 = st.columns(3)

        x1.metric(
            "Rows",
            f"{len(df):,}"
        )

        x2.metric(
            "Columns",
            len(df.columns)
        )

        x3.metric(
            "Missing Cells",
            int(df.isna().sum().sum())
        )

        tab1, tab2, tab3 = st.tabs([
            "Salary Distribution",
            "Feature Relationships",
            "Data Preview"
        ])

        with tab1:

            fig = px.histogram(
                df,
                x="EstimatedSalary",
                nbins=35,
                title="EstimatedSalary Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with tab2:

            numeric = df.select_dtypes(
                include=np.number
            )

            fig = px.imshow(
                numeric.corr(),
                text_auto=".2f",
                aspect="auto",
                title="Numeric Feature Correlation"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with tab3:

            st.dataframe(
                df.head(100),
                use_container_width=True
            )


# =====================================================
# 6. ANN MODEL LABORATORY
# =====================================================

elif page == "Model Laboratory":

    st.subheader("ANN Architecture Laboratory")

    if report.get("models"):

        rows = []

        for name, values in report["models"].items():

            rows.append({
                "Architecture": name,
                **values
            })

        table = pd.DataFrame(rows)

        st.dataframe(
            table,
            use_container_width=True
        )

        if {"Architecture", "MAE"}.issubset(table.columns):

            fig = px.bar(
                table,
                x="Architecture",
                y="MAE",
                title="Test MAE by Architecture"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "Architecture comparison metrics will appear "
            "after training."
        )

    st.markdown("### Available Architectures")

    st.write("""
    **Balanced ANN:** 128 → 64 → 32 neurons.

    **Deep ANN:** 256 → 128 → 64 → 32 → 16 neurons.

    **Wide ANN:** 256 → 256 → 128 neurons.
    """)

    st.caption(
        "Lower MAE and RMSE are better. Higher R² is better. "
        "Always compare the ANN against the median baseline."
    )


# =====================================================
# 7. RESEARCH NOTES
# =====================================================

elif page == "Research Notes":

    st.subheader("Research Notes & Model Interpretation")

    st.markdown("""
    ### Why regression?

    EstimatedSalary is a continuous numerical target.

    ### Why a linear output?

    A linear output neuron can predict a continuous value
    rather than a class probability.

    ### What does MAE mean?

    Mean Absolute Error is the average absolute difference
    between actual and predicted salary, in the target's
    currency units.

    ### Why can RMSE exceed MAE?

    RMSE squares errors before averaging. This gives
    larger errors more influence.

    ### What does R² mean?

    R² measures performance relative to predicting the
    mean target value. It can be negative.

    ### Limitations

    This classroom dataset may not contain enough
    information to predict salary accurately. It does not
    establish verified income, representativeness, fairness,
    or suitability for financial decisions.
    """)

    if report:

        st.markdown("### Model Evaluation Report")

        st.json(report)