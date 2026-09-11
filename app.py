
import streamlit as st
import pandas as pd
import joblib
import numpy as np


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

model = joblib.load("models/loan_data_model.pkl")
scaler = joblib.load("models/scaler.pkl")


# =========================================================
# MAIN TITLE
# =========================================================

st.title("🏦 AI Loan Approval Prediction System")

st.write(
    "Predict whether a loan application is likely to be approved "
    "using Machine Learning."
)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    [
        "Single Prediction",
        "Batch CSV Prediction",
        "Analytics",
        "About"
    ]
)


# =========================================================
# 1. SINGLE PREDICTION
# =========================================================

if page == "Single Prediction":

    st.header("📝 Single Loan Prediction")

    st.write(
        "Enter the applicant information below to predict loan approval."
    )

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        [0, 1, 2, 3]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0
    )

    co_income = st.number_input(
        "Co-applicant Income",
        min_value=0.0,
        value=0.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=100.0
    )

    loan_term = st.number_input(
        "Loan Term",
        min_value=0.0,
        value=360.0
    )

    credit_history = st.selectbox(
        "Credit History",
        [0, 1]
    )

    property_area = st.selectbox(
        "Property Area",
        [
            "Urban",
            "Semiurban",
            "Rural"
        ]
    )


    # -----------------------------------------------------
    # CONVERT CATEGORICAL VALUES
    # -----------------------------------------------------

    gender_value = 1 if gender == "Male" else 0

    married_value = 1 if married == "Yes" else 0

    education_value = 1 if education == "Graduate" else 0

    self_employed_value = 1 if self_employed == "Yes" else 0

    property_value = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }[property_area]


    # -----------------------------------------------------
    # CREATE ADDITIONAL FEATURES
    # -----------------------------------------------------

    total_income = income + co_income

    income_per_dependent = (
        total_income / (dependents + 1)
    )

    if total_income > 0:

        loan_income_ratio = (
            loan_amount / total_income
        )

    else:

        loan_income_ratio = 0


    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict Loan Approval",
        type="primary"
    ):

        # Create input features
        features = np.array([
            [
                gender_value,
                married_value,
                dependents,
                education_value,
                self_employed_value,
                income,
                co_income,
                loan_amount,
                loan_term,
                credit_history,
                property_value,
                total_income,
                income_per_dependent,
                loan_income_ratio
            ]
        ])


        # Scale features
        scaled = scaler.transform(features)


        # Make prediction
        prediction = model.predict(scaled)


        # Get probability
        probability = model.predict_proba(scaled)


        # Calculate confidence
        confidence = probability.max() * 100


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.subheader("📊 Prediction Result")


        if prediction[0] == 1:

            st.success(
                "✅ Loan Approved"
            )

        else:

            st.error(
                "❌ Loan Rejected"
            )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


        # -------------------------------------------------
        # RISK
        # -------------------------------------------------

        if confidence > 90:

            st.success(
                "🟢 Low Risk"
            )

        elif confidence > 75:

            st.warning(
                "🟡 Medium Risk"
            )

        else:

            st.error(
                "🔴 High Risk"
            )


        # -------------------------------------------------
        # APPLICATION SUMMARY
        # -------------------------------------------------

        st.subheader("📋 Application Summary")


        summary = pd.DataFrame({

            "Feature": [
                "Gender",
                "Married",
                "Dependents",
                "Education",
                "Self Employed",
                "Applicant Income",
                "Co-applicant Income",
                "Loan Amount",
                "Loan Term",
                "Credit History",
                "Property Area",
                "Total Income"
            ],

            "Value": [
                gender,
                married,
                dependents,
                education,
                self_employed,
                income,
                co_income,
                loan_amount,
                loan_term,
                credit_history,
                property_area,
                total_income
            ]

        })


        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# 2. BATCH CSV PREDICTION
# =========================================================

elif page == "Batch CSV Prediction":

    st.header("📂 Batch CSV Prediction")

    st.write(
        "Upload a CSV file containing multiple loan applications. "
        "The model will predict approval or rejection for every applicant."
    )


    # -----------------------------------------------------
    # FILE UPLOADER
    # -----------------------------------------------------

    uploaded = st.file_uploader(
        "Upload Loan CSV File",
        type=["csv"]
    )


    # -----------------------------------------------------
    # ONLY RUN WHEN FILE IS UPLOADED
    # -----------------------------------------------------

    if uploaded is not None:

        try:

            # Read uploaded CSV
            batch = pd.read_csv(uploaded)


            st.success(
                "✅ CSV uploaded successfully!"
            )


            # -------------------------------------------------
            # SHOW ORIGINAL DATA
            # -------------------------------------------------

            st.subheader("📄 Uploaded Data")

            st.dataframe(
                batch,
                use_container_width=True
            )


            # -------------------------------------------------
            # REQUIRED COLUMNS
            # -------------------------------------------------

            required_columns = [
                "Gender",
                "Married",
                "Dependents",
                "Education",
                "Self_Employed",
                "ApplicantIncome",
                "CoapplicantIncome",
                "LoanAmount",
                "Loan_Amount_Term",
                "Credit_History",
                "Property_Area"
            ]


            missing_columns = []

            for column in required_columns:

                if column not in batch.columns:

                    missing_columns.append(column)


            # -------------------------------------------------
            # CHECK MISSING COLUMNS
            # -------------------------------------------------

            if missing_columns:

                st.error(
                    "❌ Your CSV is missing required columns."
                )

                st.write(
                    missing_columns
                )


            else:

                # -------------------------------------------------
                # COPY DATA FOR PROCESSING
                # -------------------------------------------------

                prediction_data = batch.copy()


                # -------------------------------------------------
                # DEPENDENTS
                # -------------------------------------------------

                prediction_data["Dependents"] = (
                    prediction_data["Dependents"]
                    .astype(str)
                    .replace("3+", "3")
                )

                prediction_data["Dependents"] = pd.to_numeric(
                    prediction_data["Dependents"],
                    errors="coerce"
                )


                # -------------------------------------------------
                # GENDER
                # -------------------------------------------------

                prediction_data["Gender"] = (
                    prediction_data["Gender"]
                    .map({
                        "Male": 1,
                        "Female": 0
                    })
                )


                # -------------------------------------------------
                # MARRIED
                # -------------------------------------------------

                prediction_data["Married"] = (
                    prediction_data["Married"]
                    .map({
                        "Yes": 1,
                        "No": 0
                    })
                )


                # -------------------------------------------------
                # EDUCATION
                # -------------------------------------------------

                prediction_data["Education"] = (
                    prediction_data["Education"]
                    .map({
                        "Graduate": 1,
                        "Not Graduate": 0
                    })
                )


                # -------------------------------------------------
                # SELF EMPLOYED
                # -------------------------------------------------

                prediction_data["Self_Employed"] = (
                    prediction_data["Self_Employed"]
                    .map({
                        "Yes": 1,
                        "No": 0
                    })
                )


                # -------------------------------------------------
                # PROPERTY AREA
                # -------------------------------------------------

                prediction_data["Property_Area"] = (
                    prediction_data["Property_Area"]
                    .map({
                        "Rural": 0,
                        "Semiurban": 1,
                        "Urban": 2
                    })
                )


                # -------------------------------------------------
                # NUMERIC COLUMNS
                # -------------------------------------------------

                numeric_columns = [
                    "ApplicantIncome",
                    "CoapplicantIncome",
                    "LoanAmount",
                    "Loan_Amount_Term",
                    "Credit_History"
                ]


                for column in numeric_columns:

                    prediction_data[column] = pd.to_numeric(
                        prediction_data[column],
                        errors="coerce"
                    )


                # -------------------------------------------------
                # HANDLE MISSING VALUES
                # -------------------------------------------------

                for column in numeric_columns:

                    prediction_data[column] = (
                        prediction_data[column]
                        .fillna(
                            prediction_data[column].median()
                        )
                    )


                prediction_data["Dependents"] = (
                    prediction_data["Dependents"]
                    .fillna(0)
                )


                # -------------------------------------------------
                # HANDLE ANY REMAINING MISSING VALUES
                # -------------------------------------------------

                prediction_data = prediction_data.fillna(0)


                # -------------------------------------------------
                # TOTAL INCOME
                # -------------------------------------------------

                prediction_data["TotalIncome"] = (
                    prediction_data["ApplicantIncome"]
                    +
                    prediction_data["CoapplicantIncome"]
                )


                # -------------------------------------------------
                # INCOME PER DEPENDENT
                # -------------------------------------------------

                prediction_data["IncomePerDependent"] = (
                    prediction_data["TotalIncome"]
                    /
                    (
                        prediction_data["Dependents"] + 1
                    )
                )


                # -------------------------------------------------
                # LOAN INCOME RATIO
                # -------------------------------------------------

                prediction_data["LoanIncomeRatio"] = np.where(

                    prediction_data["TotalIncome"] > 0,

                    prediction_data["LoanAmount"]
                    /
                    prediction_data["TotalIncome"],

                    0
                )


                # -------------------------------------------------
                # SELECT MODEL FEATURES
                # -------------------------------------------------

                features_batch = prediction_data[
                    [
                        "Gender",
                        "Married",
                        "Dependents",
                        "Education",
                        "Self_Employed",
                        "ApplicantIncome",
                        "CoapplicantIncome",
                        "LoanAmount",
                        "Loan_Amount_Term",
                        "Credit_History",
                        "Property_Area",
                        "TotalIncome",
                        "IncomePerDependent",
                        "LoanIncomeRatio"
                    ]
                ]


                # -------------------------------------------------
                # CONVERT TO NUMPY
                # -------------------------------------------------

                features_batch = features_batch.values


                # -------------------------------------------------
                # SCALE
                # -------------------------------------------------

                scaled_batch = scaler.transform(
                    features_batch
                )


                # -------------------------------------------------
                # PREDICTION
                # -------------------------------------------------

                predictions = model.predict(
                    scaled_batch
                )


                # -------------------------------------------------
                # PROBABILITY
                # -------------------------------------------------

                probabilities = model.predict_proba(
                    scaled_batch
                )


                # -------------------------------------------------
                # ADD PREDICTIONS TO ORIGINAL DATA
                # -------------------------------------------------

                batch["Prediction"] = np.where(
                    predictions == 1,
                    "Approved",
                    "Rejected"
                )


                # -------------------------------------------------
                # ADD CONFIDENCE
                # -------------------------------------------------

                batch["Confidence"] = (
                    probabilities.max(axis=1) * 100
                ).round(2)


                # -------------------------------------------------
                # SHOW RESULTS
                # -------------------------------------------------

                st.subheader(
                    "🔮 Batch Prediction Results"
                )


                st.dataframe(
                    batch,
                    use_container_width=True
                )


                # -------------------------------------------------
                # APPROVED / REJECTED COUNTS
                # -------------------------------------------------

                approved_count = int(
                    (predictions == 1).sum()
                )

                rejected_count = int(
                    (predictions == 0).sum()
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Total Applications",
                        len(batch)
                    )


                with col2:

                    st.metric(
                        "✅ Approved",
                        approved_count
                    )


                with col3:

                    st.metric(
                        "❌ Rejected",
                        rejected_count
                    )


                # -------------------------------------------------
                # DOWNLOAD
                # -------------------------------------------------

                csv = batch.to_csv(
                    index=False
                )


                st.download_button(

                    label="⬇️ Download Predictions CSV",

                    data=csv,

                    file_name="loan_predictions.csv",

                    mime="text/csv"

                )


        except Exception as e:

            st.error(
                f"❌ Error processing CSV: {e}"
            )


# =========================================================
# 3. ANALYTICS
# =========================================================

elif page == "Analytics":

    st.header("📊 Loan Dataset Analytics")

    st.write(
        "Explore the loan dataset used by the application."
    )


    try:

        # -------------------------------------------------
        # LOAD DATASET
        # -------------------------------------------------

        df = pd.read_csv(
            "dataset/loan_cleaned.csv"
        )


        # -------------------------------------------------
        # DATA PREVIEW
        # -------------------------------------------------

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


        # -------------------------------------------------
        # DATASET METRICS
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Applications",
                len(df)
            )


        with col2:

            st.metric(
                "Total Columns",
                len(df.columns)
            )


        with col3:

            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )


        # -------------------------------------------------
        # APPLICANT INCOME
        # -------------------------------------------------

        if "ApplicantIncome" in df.columns:

            st.subheader(
                "💰 Applicant Income"
            )

            st.bar_chart(
                df["ApplicantIncome"]
            )


        # -------------------------------------------------
        # LOAN AMOUNT
        # -------------------------------------------------

        if "LoanAmount" in df.columns:

            st.subheader(
                "🏦 Loan Amount"
            )

            st.line_chart(
                df["LoanAmount"]
            )


        # -------------------------------------------------
        # LOAN STATUS
        # -------------------------------------------------

        if "Loan_Status" in df.columns:

            st.subheader(
                "📈 Loan Approval Distribution"
            )

            status_counts = (
                df["Loan_Status"]
                .value_counts()
            )

            st.bar_chart(
                status_counts
            )


        # -------------------------------------------------
        # CREDIT HISTORY
        # -------------------------------------------------

        if "Credit_History" in df.columns:

            st.subheader(
                "📊 Credit History Distribution"
            )

            credit_counts = (
                df["Credit_History"]
                .value_counts()
            )

            st.bar_chart(
                credit_counts
            )


        # -------------------------------------------------
        # PROPERTY AREA
        # -------------------------------------------------

        if "Property_Area" in df.columns:

            st.subheader(
                "🏠 Property Area Distribution"
            )

            property_counts = (
                df["Property_Area"]
                .value_counts()
            )

            st.bar_chart(
                property_counts
            )


    except FileNotFoundError:

        st.error(
            "❌ Dataset file not found."
        )

        st.info(
            "Please make sure this file exists:"
        )

        st.code(
            "dataset/loan_cleaned.csv"
        )


# =========================================================
# 4. ABOUT
# =========================================================

elif page == "About":

    st.header("ℹ️ About This Project")


    st.write(
        """
        This AI Loan Approval Prediction System uses
        Machine Learning to predict whether a loan
        application is likely to be approved or rejected.
        """
    )


    # -----------------------------------------------------
    # MACHINE LEARNING ALGORITHMS
    # -----------------------------------------------------

    st.subheader(
        "🤖 Machine Learning Algorithms"
    )

    st.write(
        """
        • Logistic Regression

        • Decision Tree

        • Random Forest

        • XGBoost
        """
    )


    # -----------------------------------------------------
    # TECHNOLOGIES
    # -----------------------------------------------------

    st.subheader(
        "🛠️ Technologies Used"
    )

    st.write(
        """
        • Python

        • NumPy

        • Pandas

        • Scikit-Learn

        • Streamlit

        • Machine Learning
        """
    )


    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    st.subheader(
        "✨ Application Features"
    )

    st.write(
        """
        • Single loan prediction

        • Batch CSV prediction

        • Prediction confidence

        • Risk level

        • Loan dataset analytics

        • CSV prediction download

        • Interactive Streamlit interface
        """
    )


    # -----------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------

    st.info(
        "⚠️ This application is intended for educational "
        "and demonstration purposes. It should not be used "
        "as the sole basis for real financial decisions."
    )


