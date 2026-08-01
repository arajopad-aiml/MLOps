import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.set_page_config(layout="wide")
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase a newly introduced Wellness Tourism Package
based on their details and interaction data. Enter the customer information below to get a prediction.
""")

# Create columns for centering the input widgets
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    st.header("Customer Input Features")

    # Use two internal columns for better layout within the centered section
    col_a, col_b = st.columns(2)

    with col_a:
        # Numerical features as number_input
        age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=60, value=15, step=1)
        number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2, step=1)
        number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
        number_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=3, step=1)
        # Categorical features as selectbox
        type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
        occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
        gender = st.selectbox("Gender", ['Male', 'Female'])

    with col_b:
        pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3, step=1)
        number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=3, value=0, step=1)
        monthly_income = st.number_input("Monthly Income", min_value=0.0, value=25000.0, step=100.0)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
        preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
        marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])

    # Checkboxes for binary features
    passport = st.checkbox("Has Passport?")
    own_car = st.checkbox("Owns Car?")
    designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])

    # Convert boolean checkboxes to integers
    passport_val = 1 if passport else 0
    own_car_val = 1 if own_car else 0

    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport_val,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car_val,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])


    # Prediction button in the centered column
    if st.button("Predict Purchase"):
        prediction_proba = model.predict_proba(input_data)[0, 1]

    # Keep this consistent with train.py
    classification_threshold = 0.45
    prediction = int(prediction_proba >= classification_threshold)

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success(
            f"The model predicts: **Customer WILL purchase the package!** "
            f"(Probability: {prediction_proba:.2%})"
        )
    else:
        st.warning(
            f"The model predicts: **Customer will NOT purchase the package.** "
            f"(Probability: {prediction_proba:.2%})"
        )

    # Prediction button in the centered column
    if st.button("Predict Purchase"):
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[:, 1][0] # Probability of purchasing

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success(f"The model predicts: **Customer WILL purchase the package!** (Probability: {prediction_proba:.2f})")
        else:
            st.warning(f"The model predicts: **Customer will NOT purchase the package.** (Probability: {prediction_proba:.2f})")

st.write("""
### Feature Information:
*   **Age**: Age of the customer.
*   **TypeofContact**: How the customer was contacted.
*   **CityTier**: City category (Tier 1 is highest).
*   **DurationOfPitch**: Length of the sales pitch.
*   **Occupation**: Customer's job type.
*   **Gender**: Customer's gender.
*   **NumberOfPersonVisiting**: Total number of people in the travel group.
*   **NumberOfFollowups**: Salesperson follow-ups.
*   **ProductPitched**: Type of tourism product offered.
*   **PreferredPropertyStar**: Customer's preferred hotel star rating.
*   **MaritalStatus**: Customer's marital status.
*   **NumberOfTrips**: Annual number of trips taken.
*   **Passport**: Whether the customer has a valid passport.
*   **PitchSatisfactionScore**: Customer satisfaction with the pitch.
*   **OwnCar**: Whether the customer owns a car.
*   **NumberOfChildrenVisiting**: Number of children in the group.
*   **Designation**: Customer's job designation.
*   **MonthlyIncome**: Customer's gross monthly income.
""")
