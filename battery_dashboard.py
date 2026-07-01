import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Title
st.title("🔋 Smart Battery Management System")

# Load data
data = pd.read_csv(
    r"D:\Projects\Smart-Battery-Management-System\dataset\battery_data.csv"
)

# Train AI model
X = data[['Voltage', 'Current', 'Temperature']]
y = data['SOH']

model = RandomForestRegressor()
model.fit(X, y)

# User input
st.header("Battery Parameters")

voltage = st.number_input("Voltage (V)",
    min_value=3.5,
    max_value=4.2,
    value=3.8,
    step=0.01)
current = st.number_input(
    "Current (A)",
    min_value=1.0,
    max_value=4.0,
    value=2.5,
    step=0.01
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=20,
    max_value=50,
    value=35,
    step=1
)
charge = st.number_input("Charge Time", value=120)
discharge = st.number_input("Discharge Time", value=90)
resistance = st.number_input("Internal Resistance", value=0.05)
capacity = st.number_input("Capacity", value=2.8)
humidity = st.number_input("Ambient Humidity", value=60)
crate = st.number_input("C-Rate", value=1.0)
# Prediction
input_data = pd.DataFrame({
    'Voltage': [voltage],
    'Current': [current],
    'Temperature': [temperature]
})

prediction = model.predict(input_data)

st.success(f"Predicted Battery Health: {prediction[0]:.2f}%")
#Battery Health(SOH) vs Cycle graph
st.subheader("📈 Battery Health Degradation")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(data['Cycle'], data['SOH'],
        color='blue',
        marker='o')

ax.set_xlabel("Cycle Number")
ax.set_ylabel("SOH (%)")
ax.set_title("Battery State of Health vs Cycle")
ax.grid(True)

st.pyplot(fig)

#Capacity vs Cycle graph
st.subheader("🔋 Battery Capacity Degradation")

fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(data['Cycle'],
         data['Capacity'],
         color='green')

ax2.set_xlabel("Cycle Number")
ax2.set_ylabel("Capacity")
ax2.set_title("Battery Capacity vs Cycle")
ax2.grid(True)

st.pyplot(fig2)

#Temperature vs Cycle graph

st.subheader("🌡 Battery Temperature Trend")

fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.plot(data['Cycle'],
         data['Temperature'],
         color='red')

ax3.set_xlabel("Cycle Number")
ax3.set_ylabel("Temperature (°C)")
ax3.set_title("Battery Temperature vs Cycle")
ax3.grid(True)

st.pyplot(fig3)
