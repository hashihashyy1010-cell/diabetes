import streamlit as st
import pandas as pd
import joblib

# 1. Page Styling
st.set_page_config(page_title="Diabetes", layout="centered")

st.title("Diabetes")
st.write("Enter the total result.")

try:
    # 2. Load the trained model
    model = joblib.load('Diabetes.pkl')


    # 3. Create a Layout with Columns for User Input
    col1, col2, col3 = st.columns(3)

    with col1:
        Glucose = st.number_input("Glucose", min_value=0.0, max_value=500.0, value=0.0)

    with col2:
        BMI = st.number_input("BMI", min_value=0.0, max_value=100.0, value=37.8)

    with col3:
        Age = st.number_input("Age", min_value=0.0, max_value=200.0, value=69.2)

    # 4. Create a 'Predict' button
    if st.button("Calculate Prediction"):
        # Create a DataFrame from the dynamic user input
        user_input = pd.DataFrame([{
            'Glucose': Glucose,
            'BMI': BMI,
            'Age': Age
        }])

        # Get prediction
        prediction = model.predict(user_input)

        # 5. Display Result in a nice box
        st.divider()
        st.subheader("Results")
        st.metric(label="Diabetes", value=f"{prediction[0]:.2f}")

        # Show how the input compares
        st.bar_chart(user_input.T)

except Exception as e:
    st.error(f"Model Error: {e}")
