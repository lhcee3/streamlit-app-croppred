import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
import streamlit as st


df = pd.read_excel('Advanced_Crop_Prediction_and_Analysis_2024.xlsx', sheet_name='Sheet1')


df['Month'] = df['Month'].astype(str)
df['Soil_Type'] = df['Soil_Type'].astype(str)


label_encoders = {}
for column in ['Crop', 'Month', 'Soil_Type', 'Recommended_Fertilizers', 'Storage_Conditions']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


X = df[['Month', 'Soil_Type', 'Nitrogen (%)', 'Phosphorus (%)', 'Potassium (%)',
        'Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)', 'Weather_Deviation_Index']]

y_crop = df['Crop']  
y_price = df['Price_Per_Ton (₹)']  


X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X, y_crop, test_size=0.2, random_state=42)


X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.2, random_state=42)


crop_model = DecisionTreeClassifier(random_state=42)
crop_model.fit(X_train_crop, y_train_crop)


price_model = RandomForestRegressor(random_state=42)
price_model.fit(X_train_price, y_train_price)


crop_accuracy = crop_model.score(X_test_crop, y_test_crop)
price_mse = mean_squared_error(y_test_price, price_model.predict(X_test_price))
print(f"Crop Prediction Accuracy: {crop_accuracy * 100:.2f}%")
print(f"Price Prediction MSE: {price_mse:.2f}")


def predict_crop_and_price(month, temperature, rainfall, soil_type):
    
    if month not in label_encoders['Month'].classes_:
        month_encoded = label_encoders['Month'].transform([label_encoders['Month'].classes_[0]])[0]
    else:
        month_encoded = label_encoders['Month'].transform([month])[0]
    
    if soil_type not in label_encoders['Soil_Type'].classes_:
        soil_type_encoded = label_encoders['Soil_Type'].transform([label_encoders['Soil_Type'].classes_[0]])[0]
    else:
        soil_type_encoded = label_encoders['Soil_Type'].transform([soil_type])[0]

    
    input_data = pd.DataFrame({
        'Month': [month_encoded],
        'Soil_Type': [soil_type_encoded],
        'Nitrogen (%)': [2.0],
        'Phosphorus (%)': [1.0],
        'Potassium (%)': [1.5],
        'Temperature (°C)': [temperature],
        'Humidity (%)': [70.0],
        'Rainfall (mm)': [rainfall],
        'Weather_Deviation_Index': [0.0]
    })

    
    predicted_crop = crop_model.predict(input_data)[0]
    predicted_crop_name = label_encoders['Crop'].inverse_transform([predicted_crop])[0]

    
    predicted_price = price_model.predict(input_data)[0]

    return predicted_crop_name, predicted_price

# Streamlit app
st.title("🌽Crop and Price Prediction")


st.write(f"Crop Prediction Accuracy: {crop_accuracy * 100:.2f}%")
st.write(f"Price Prediction MSE: {price_mse:.2f}")


month_input = st.selectbox("Select the month:", df['Month'].unique())
soil_type_input = st.selectbox("Select the soil type:", df['Soil_Type'].unique())
temperature_input = st.slider("Select the temperature (°C):", min_value=0.0, max_value=50.0, value=25.0)
rainfall_input = st.slider("Select the rainfall (mm):", min_value=0.0, max_value=500.0, value=100.0)

if st.button("Predict"):
    predicted_crop, predicted_price = predict_crop_and_price(month_input, temperature_input, rainfall_input, soil_type_input)
    if predicted_crop and predicted_price:
        st.write(f"For the month of {month_input}, with soil type '{soil_type_input}', temperature {temperature_input}°C, and rainfall {rainfall_input}mm, the best crop is {predicted_crop} with an estimated market price of ₹{predicted_price:.2f} per kg.")