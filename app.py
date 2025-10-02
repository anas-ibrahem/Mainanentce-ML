import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Predictive Maintenance App",
    page_icon="🤖",
    layout="wide"
)

# --- MODEL AND SCALER LOADING ---

# Use a cache decorator to load the model and scalers only once
@st.cache_resource
def load_model_and_scalers():
    """
    Loads the trained model and fits the scalers on the original dataset
    based on the new feature requirements.
    """
    try:
        model = joblib.load('model.joblib')
        # Load the raw data to fit the scalers as they weren't saved
        df_raw = pd.read_csv("ai4i2020.csv")

        # --- MODIFIED FEATURE ENGINEERING FOR SCALER FITTING ---
        # 1. Create engineered features
        df_raw['Power [W]'] = df_raw['Torque [Nm]'] * df_raw['Rotational speed [rpm]']
        df_raw['Delta_T [K]'] = df_raw['Process temperature [K]'] - df_raw['Air temperature [K]']
        
        # 2. Apply one-hot encoding
        df_raw = pd.get_dummies(df_raw, columns=['Type'], prefix='Type', dtype=int)

        # 3. Define features for each scaler (one-hot columns are not scaled)
        robust_features = ['Power [W]']
        standard_features = ['Tool wear [min]', 'Delta_T [K]']

        # Fit RobustScaler
        robust_scaler = RobustScaler()
        robust_scaler.fit(df_raw[robust_features])

        # Fit StandardScaler
        standard_scaler = StandardScaler()
        standard_scaler.fit(df_raw[standard_features])

        return model, robust_scaler, standard_scaler
    except FileNotFoundError:
        return None, None, None

model, robust_scaler, standard_scaler = load_model_and_scalers()

# --- APP UI ---
st.title("⚙️ Predictive Maintenance Dashboard")
st.markdown("Predict machine failure based on sensor data. This model uses engineered features and one-hot encoding.")

if model is None or robust_scaler is None or standard_scaler is None:
    st.error(
        "**Error:** `model.joblib` or `ai4i2020.csv` not found. "
        "Please make sure both files are in the same directory as `app.py`."
    )
else:
    # --- SIDEBAR FOR USER INPUT ---
    st.sidebar.header("Machine Sensor Inputs")
    st.sidebar.markdown("Adjust the values below to get a failure prediction.")

    # --- MODIFICATION START ---
    # Changed st.slider to st.number_input to allow direct text entry.
    # Using named arguments for clarity.
    air_temp = st.sidebar.number_input(
        'Air Temperature [K]', 
        min_value=295.0, 
        max_value=305.0, 
        value=300.1, 
        step=0.1,
        format="%.1f"
    )
    process_temp = st.sidebar.number_input(
        'Process Temperature [K]', 
        min_value=305.0, 
        max_value=314.0, 
        value=310.1, 
        step=0.1,
        format="%.1f"
    )
    rotational_speed = st.sidebar.number_input(
        'Rotational Speed [rpm]', 
        min_value=1150, 
        max_value=2900, 
        value=1503, 
        step=1
    )
    torque = st.sidebar.number_input(
        'Torque [Nm]', 
        min_value=3.0, 
        max_value=80.0, 
        value=40.1, 
        step=0.1,
        format="%.1f"
    )
    tool_wear = st.sidebar.number_input(
        'Tool Wear [min]', 
        min_value=0, 
        max_value=255, 
        value=108, 
        step=1
    )
    # --- MODIFICATION END ---
    
    machine_type = st.sidebar.selectbox('Machine Type', ('L', 'M', 'H'))

    predict_button = st.sidebar.button("📊 Predict Failure", use_container_width=True)

    # --- PREDICTION LOGIC AND DISPLAY (No changes below this line) ---
    if predict_button:
        # 1. Create a DataFrame from user inputs
        input_data = {
            'Air temperature [K]': [air_temp],
            'Process temperature [K]': [process_temp],
            'Rotational speed [rpm]': [rotational_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear],
            'Type': [machine_type]
        }
        input_df = pd.DataFrame(input_data)

        # 2. Apply the same feature engineering
        input_df['Power [W]'] = input_df['Torque [Nm]'] * input_df['Rotational speed [rpm]']
        input_df['Delta_T [K]'] = input_df['Process temperature [K]'] - input_df['Air temperature [K]']
        
        input_df['Type_H'] = 1 if machine_type == 'H' else 0
        input_df['Type_L'] = 1 if machine_type == 'L' else 0
        input_df['Type_M'] = 1 if machine_type == 'M' else 0

        # 3. Apply scaling only to the relevant columns
        robust_features = ['Power [W]']
        standard_features = ['Tool wear [min]', 'Delta_T [K]']

        input_df[robust_features] = robust_scaler.transform(input_df[robust_features])
        input_df[standard_features] = standard_scaler.transform(input_df[standard_features])

        # 4. Ensure feature order matches the model's training order EXACTLY
        feature_order = [
            'Tool wear [min]', 'Type_H', 'Type_L', 'Type_M', 'Power [W]', 'Delta_T [K]'
        ]
        processed_input = input_df[feature_order]

        # 5. Make prediction and get probabilities
        prediction = model.predict(processed_input)[0]
        probabilities = model.predict_proba(processed_input)[0]

        # --- DISPLAY RESULTS ---
        st.subheader("Prediction Result")

        failure_map = {
            0: ('No Failure', '✅'),
            1: ('Heat Dissipation Failure (HDF)', '🔥'),
            2: ('Power Failure (PWF)', '⚡'),
            3: ('Overstrain Failure (OSF)', '❗'),
            4: ('Tool Wear Failure (TWF)', '🛠️'),
            5: ('Random Failure (RNF)', '❓') 
        }
        
        predicted_label, icon = failure_map.get(prediction, ("Unknown", "🤷"))
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f'<p style="font-size: 80px; text-align: center;">{icon}</p>', unsafe_allow_html=True)
        with col2:
            if prediction == 0:
                st.success(f"**Status:** {predicted_label}")
                st.write("The model predicts that the machine will **not** fail under these conditions.")
            else:
                st.error(f"**Status:** {predicted_label}")
                st.write("The model predicts a **high probability of machine failure**.")

        st.subheader("Failure Type Probabilities")
        st.write("This chart shows the model's confidence for each potential failure type.")
        
        prob_df = pd.DataFrame({
            'Failure Type': [failure_map[i][0] for i in range(len(probabilities))],
            'Probability': probabilities
        })

        prob_df['Color'] = ['#d9534f' if p > 0.5 else '#5bc0de' for p in prob_df['Probability']]
        prob_df.loc[prob_df['Failure Type'] == 'No Failure', 'Color'] = '#5cb85c'

        prob_df = prob_df.set_index('Failure Type')

        st.bar_chart(prob_df, y='Probability', color='Color')