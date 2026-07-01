import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv(r"D:\Projects\Smart-Battery-Management-System\dataset\battery_data.csv")

# Input and output
X = data[['Voltage',
          'Current',
          'Temperature',
          'ChargeTime',
          'DischargeTime',
          'InternalResistance',
          'Capacity',
          'AmbientHumidity',
          'C_Rate']]
y = data['SOH']

# Train model
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Predict battery health
new_data = pd.DataFrame({
    'Voltage': [3.8],
    'Current': [2.5],
    'Temperature': [35],
    'ChargeTime': [120],
    'DischargeTime': [90],
    'InternalResistance': [0.05],
    'Capacity': [2.8],
    'AmbientHumidity': [60],
    'C_Rate': [1.0]
})

prediction = model.predict(new_data)

print("Predicted Battery Health:", prediction[0], "%")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(data['Cycle'], data['SOH'])
plt.xlabel('Cycle')
plt.ylabel('SOH (%)')
plt.title('Battery Health Degradation')
plt.grid(True)
plt.show()
