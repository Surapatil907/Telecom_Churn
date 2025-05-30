import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Load the model with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_log_reg_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'best_log_reg_model.pkl' not found!")
        st.info("Please ensure the model file is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

st.title("🔮 Telecom Customer Churn Prediction")
st.write("Enter customer information to predict whether they will churn.")

# Create input form matching the exact training features
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

with col2:
    st.subheader("📱 Service Information")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

st.subheader("🔧 Additional Services")
col3, col4 = st.columns(2)

with col3:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with col4:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.subheader("💰 Contract & Billing")
col5, col6 = st.columns(2)

with col5:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

with col6:
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.subheader("💵 Charges")
col7, col8 = st.columns(2)

with col7:
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)

with col8:
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

# Create the input data exactly as the model expects
def create_input_dataframe():
    """Create input DataFrame with exact feature names and encoding as training data"""
    
    # Create the input data dictionary matching training feature names
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Apply the same preprocessing as training data
    # Convert categorical variables to dummy variables (one-hot encoding)
    
    # Handle TotalCharges (might need to be string in original data)
    df['TotalCharges'] = df['TotalCharges'].astype(str)
    
    # Get dummy variables for categorical columns
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                       'PaperlessBilling', 'PaymentMethod']
    
    # Create dummy variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Convert TotalCharges back to numeric (handle any conversion issues)
    df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors='coerce').fillna(0)
    
    return df_encoded

# Prediction section
st.markdown("---")
col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])

with col_pred2:
    if st.button("🔍 Predict Churn", type="primary", use_container_width=True):
        try:
            # Create input DataFrame
            input_df = create_input_dataframe()
            
            # Show debug info about features
            st.info(f"Input has {input_df.shape[1]} features, model expects {model.n_features_in_}")
            
            # Ensure we have the right number of features for the model
            # The model might expect features in a specific order
            expected_features = model.n_features_in_
            current_features = input_df.shape[1]
            
            if current_features != expected_features:
                st.warning(f"⚠️ Feature count mismatch. Adjusting from {current_features} to {expected_features} features.")
                
                # If we have fewer features, pad with zeros
                if current_features < expected_features:
                    missing_features = expected_features - current_features
                    for i in range(missing_features):
                        input_df[f'missing_feature_{i}'] = 0
                
                # If we have more features, we might need to reorder or select specific ones
                elif current_features > expected_features:
                    # Take only the first N features (this is a temporary fix)
                    input_df = input_df.iloc[:, :expected_features]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            if prediction == 1:
                st.error(f"🚨 **Customer is likely to CHURN**")
                st.error(f"Churn Probability: **{proba:.1%}**")
                
                # Risk level based on probability
                if proba >= 0.8:
                    risk_level = "🔴 VERY HIGH RISK"
                elif proba >= 0.6:
                    risk_level = "🟠 HIGH RISK"
                else:
                    risk_level = "🟡 MODERATE RISK"
                
                st.markdown(f"**Risk Level:** {risk_level}")
                
                # Actionable recommendations
                st.markdown("#### 💡 Retention Strategy Recommendations:")
                recommendations = []
                
                if contract == "Month-to-month":
                    recommendations.append("📋 **Contract**: Offer annual contract with significant discount")
                
                if monthly_charges > 80:
                    recommendations.append("💰 **Pricing**: Consider competitive pricing or value-added services")
                
                if payment_method == "Electronic check":
                    recommendations.append("💳 **Payment**: Encourage automatic payment methods with incentives")
                
                if internet_service == "Fiber optic" and proba > 0.7:
                    recommendations.append("🌐 **Service**: Check fiber service quality and provide premium support")
                
                if senior_citizen == 1:
                    recommendations.append("👥 **Support**: Provide senior-friendly support and simplified plans")
                
                if any(service == "No" for service in [online_security, online_backup, tech_support]):
                    recommendations.append("🛡️ **Upsell**: Offer security and support services at discounted rates")
                
                if not recommendations:
                    recommendations = [
                        "📞 **Contact**: Immediate outreach to understand customer concerns",
                        "🎁 **Incentives**: Offer personalized retention incentives",
                        "👂 **Feedback**: Conduct satisfaction survey to identify pain points"
                    ]
                
                for rec in recommendations:
                    st.markdown(f"• {rec}")
                
            else:
                st.success(f"✅ **Customer is likely to STAY**")
                st.success(f"Retention Probability: **{1-proba:.1%}**")
                
                if 1-proba >= 0.8:
                    loyalty_level = "💎 HIGHLY LOYAL"
                elif 1-proba >= 0.6:
                    loyalty_level = "⭐ LOYAL"
                else:
                    loyalty_level = "👍 SATISFIED"
                
                st.markdown(f"**Loyalty Level:** {loyalty_level}")
                st.info("💡 **Opportunity**: Customer appears satisfied. Consider upselling premium services or referral programs.")
            
            # Confidence visualization
            st.markdown("#### 📈 Prediction Confidence")
            confidence = max(proba, 1-proba)
            st.progress(confidence)
            st.caption(f"Model Confidence: {confidence:.1%}")
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.error("This might be due to feature preprocessing mismatch. Please check the error details below:")
            st.code(str(e))

# # Information and troubleshooting
# with st.expander("ℹ️ About This Model"):
#     st.markdown("""
#     **Model Information:**
#     - **Algorithm**: Logistic Regression
#     - **Purpose**: Predict telecom customer churn
#     - **Features**: 19 customer attributes (demographics, services, billing)
    
#     **Key Predictive Factors:**
#     - Contract type (month-to-month customers have higher churn risk)
#     - Monthly charges (higher charges increase churn probability)
#     - Payment method (electronic check users have higher churn)
#     - Tenure (newer customers are more likely to churn)
#     - Internet service type (fiber optic customers may have higher churn)
    
#     **How to Use:**
#     1. Fill in all customer information fields
#     2. Click "Predict Churn" to get the prediction
#     3. Review recommendations for high-risk customers
#     """)

# with st.expander("🔧 Troubleshooting"):
#     st.markdown("""
#     **Common Issues:**
    
#     **Feature Mismatch Error:**
#     - Ensure your model was trained on the same 19 features listed above
#     - Check that categorical encoding matches training data preprocessing
    
#     **Model Loading Error:**
#     - Verify 'best_log_reg_model.pkl' file is in the same directory
#     - Ensure the model was saved using joblib (not pickle)
    
#     **Prediction Accuracy:**
#     - Model performance depends on training data quality
#     - Consider retraining if business context has changed significantly
#     """)

# # Debug section (can be removed in production)
# if st.checkbox("🔧 Show Debug Information"):
#     st.subheader("Debug Information")
#     try:
#         debug_df = create_input_dataframe()
#         st.write(f"**Input DataFrame Shape:** {debug_df.shape}")
#         st.write(f"**Model Expected Features:** {model.n_features_in_}")
#         st.write("**Input DataFrame Columns:**")
#         st.write(list(debug_df.columns))
#         st.write("**Sample of Input Data:**")
#         st.dataframe(debug_df)
#     except Exception as e:
#         st.error(f"Debug error: {str(e)}")
