import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load the neural network model
model_nn = tf.keras.models.load_model('model_nn.h5')

# Load other models
with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

# Load the encoders
with open('onehot_encoder_cabin.pkl', 'rb') as file:
    onehot_encoder_cabin = pickle.load(file)

with open('onehot_encoder_travellertype.pkl', 'rb') as file:
    onehot_encoder_travellertype = pickle.load(file)

# App Title
st.title('Passenger Referral Predictor')

# User Inputs
st.subheader("Rate Your Experience (0 = Worst, 5 = Best)")

seat_comfort = st.slider('Seat Comfort', 0, 5)
cabin_service = st.slider('Cabin Service', 0, 5)
food_service = st.slider('Food & Beverage', 0, 5)
entertainment = st.slider('Entertainment', 0, 5)
ground_service = st.slider('Ground Service', 0, 5)
value_for_money = st.slider('Value for Money', 0, 5)

# Categorical Inputs
traveller_type = st.selectbox('Traveller Type', onehot_encoder_travellertype.categories_[0])
cabin_type = st.selectbox('Cabin Type', onehot_encoder_cabin.categories_[0])

# Prepare input data
input_data = pd.DataFrame({
    'seat_comfort': [seat_comfort],
    'cabin_service': [cabin_service],
    'food_bev': [food_service],
    'entertainment': [entertainment],
    'ground_service': [ground_service],
    'value_for_money': [value_for_money]
})

# One-Hot Encode Traveller Type
travellertype_encoded = onehot_encoder_travellertype.transform(np.array([[traveller_type]])).toarray()
travellertype_encoded_df = pd.DataFrame(travellertype_encoded,
                                        columns=onehot_encoder_travellertype.get_feature_names_out(['traveller_type']))

# One-Hot Encode Cabin
cabin_encoded = onehot_encoder_cabin.transform(np.array([[cabin_type]])).toarray()
cabin_encoded_df = pd.DataFrame(cabin_encoded, columns=onehot_encoder_cabin.get_feature_names_out(['cabin']))

# Combine Encoded Features
input_df = pd.concat([input_data, travellertype_encoded_df, cabin_encoded_df], axis=1)

# Get the feature names from the first model
if isinstance(models, list):
    first_model = models[0]  # If models is a list, pick the first one
else:
    first_model = next(iter(models.values()))  # If models is a dictionary, get the first model

# Extract the list of feature names used in training
model_features = list(first_model.feature_names_in_)
input_features = list(input_df.columns)

# Find missing features manually (features in the model but not in input_df)
missing_cols = []
for feature in model_features:
    if feature not in input_features:
        missing_cols.append(feature)

# Add missing features to input_df with default value 0
for col in missing_cols:
    input_df[col] = 0  # Assigning missing columns a default value of 0

# Reorder input_df columns to match model training data order
input_df = input_df[model_features]


# Function to predict using multiple models
def prediction(models_ml, input_df):
    pred_score = []  # Initialize outside the loop

    for model in models_ml:  # Loop through models
        score = model.predict(input_df)  # Get prediction
        pred_score.append(int(score[0]))  # Convert prediction to int and store

    # Return the most frequent prediction (majority vote)
    final_prediction = max(set(pred_score), key=pred_score.count)
    return final_prediction


# Prediction Button
if st.button('Predict Recommendation'):
    pred_nn = model_nn.predict(input_df)  # Neural Network Prediction
    pred_ml = prediction(models, input_df)  # Machine Learning Model Prediction

    # Convert Neural Network Output to Binary (Threshold: 0.5)
    pred_nn_binary = (pred_nn >= 0.5).astype(int)[0][0]

    # Final Decision: Use majority voting
    if pred_nn_binary == 1 and pred_ml == 1:
        st.success("Customer has Recommended the Airline! ")
    else:
        st.error("Customer did not Recommend the Airline. ")

# Footer
st.markdown("---")
st.markdown("Developed with Streamlit & TensorFlow")
