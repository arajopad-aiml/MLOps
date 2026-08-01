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

st.sidebar.header("Customer Input Features")

# Create input widgets for each feature dynamically based on the X DataFrame structure
# Numerical features
age = st.sidebar.slider("Age", min_value=18, max_value=80, value=35)
duration_of_pitch = st.sidebar.slider("Duration of Pitch (minutes)", min_value=5, max_value=60, value=15)
number_of_person_visiting = st.sidebar.slider("Number of Persons Visiting", min_value=1, max_value=5, value=2)
number_of_followups = st.sidebar.slider("Number of Follow-ups", min_value=0, max_value=10, value=3)
number_of_trips = st.sidebar.slider("Number of Trips Annually", min_value=0, max_value=20, value=3)
pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
number_of_children_visiting = st.sidebar.slider("Number of Children Visiting", min_value=0, max_value=3, value=0)
monthly_income = st.sidebar.number_input("Monthly Income", min_value=0.0, value=25000.0, step=100.0)

# Categorical and Binary features
type_of_contact = st.sidebar.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
occupation = st.sidebar.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Freelancer'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
product_pitched = st.sidebar.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
preferred_property_star = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.sidebar.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
passport = st.sidebar.checkbox("Has Passport?")
own_car = st.sidebar.checkbox("Owns Car?")
designation = st.sidebar.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])

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


if st.sidebar.button("Predict Purchase"): # Use sidebar for the button too
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
