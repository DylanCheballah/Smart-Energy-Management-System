# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score

# Defining the parameters
num_samples = 500  # Number of samples to generate
battery_capacity = 100  # Battery full capacity in arbitrary units
initial_battery = 50  # Starting battery charge level (percentage)

# Generate synthetic data
np.random.seed(42)  # For reproducibility
potentiometer_voltage = np.random.uniform(0.5, 5.0, num_samples)  # Range: 0.5V - 5V
solar_panel_voltage = np.random.uniform(0, 2.5, num_samples)  # Range: 0V - 2.5V
temperature = np.random.uniform(15, 40, num_samples)  # Range: 15°C - 40°C
light_intensity = np.random.uniform(0, 100, num_samples)  # Range: 0% - 100%

# Temperature coefficient for efficiency loss per degree Celsius above 25°C
temp_coefficient = -0.004

# Calculate efficiency and power duration based on parameters
efficiency_loss = np.maximum(temperature - 25, 0) * temp_coefficient
efficiency = 1 + efficiency_loss
battery_duration = (initial_battery * efficiency * solar_panel_voltage / potentiometer_voltage) * 2

# Create a binary target (1 if duration is sufficient, 0 if insufficient)
will_last = (battery_duration > battery_capacity / 2).astype(int)

# Create DataFrame and save features/target
data = pd.DataFrame({
    'temperature': temperature,
    'light_intensity': light_intensity,
    'potentiometer_voltage': potentiometer_voltage,
    'solar_panel_voltage': solar_panel_voltage,
    'will_last': will_last
})

X = data[['temperature', 'light_intensity', 'potentiometer_voltage', 'solar_panel_voltage']]
y = data['will_last']

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'energy_model.pkl')

# Display accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

