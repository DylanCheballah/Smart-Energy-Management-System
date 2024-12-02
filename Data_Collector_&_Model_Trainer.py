# Latest code working
# Training th machine learning model with real time data
import serial
import csv
import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Configure the Serial connection
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino's port

# CSV file path
csv_file_path = 'sensor_data.csv'

# Initialize the CSV file with headers if it doesn't already exist
if not os.path.exists(csv_file_path):  # Check if the file exists
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["temperature", "light_intensity", "potentiometer_voltage", "solar_panel_voltage"])

def read_from_serial():
    """Read and parse a line of CSV data from the Serial connection."""
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        values = line.split(',')
        if len(values) == 4:  # Ensure we have all expected values
            try:
                data = {
                    "temperature": float(values[2]),
                    "light_intensity": float(values[3]),
                    "potentiometer_voltage": float(values[0]),
                    "solar_panel_voltage": float(values[1])
                }
                return data
            except ValueError:
                print("Error parsing data:", values)
    return None

def write_to_csv(data):
    """Append only values to the CSV file under the header row."""
    with open(csv_file_path, mode='a', newline='') as file:  # Use 'a' mode to append
        writer = csv.writer(file)
        writer.writerow([
            data["temperature"],
            data["light_intensity"],
            data["potentiometer_voltage"],
            data["solar_panel_voltage"]
        ])

def train_model():
    """Train the logistic regression model on the collected data."""
    # Load the real-time data from CSV
    data = pd.read_csv(csv_file_path)

    # Ensure all columns are in numeric format
    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)  # Drop rows with NaN values, if any

    # Define parameters for training
    battery_capacity = 100  # Battery full capacity in arbitrary units
    initial_battery = 50    # Starting battery charge level (percentage)
    temp_coefficient = -0.004

    # Calculate efficiency and power duration based on real-time data
    efficiency_loss = np.maximum(data['temperature'] - 25, 0) * temp_coefficient
    efficiency = 1 + efficiency_loss

    data['battery_duration'] = (initial_battery * efficiency * data['solar_panel_voltage'] / data['potentiometer_voltage']) * 2

    # Create a binary target (1 if duration is sufficient, 0 if insufficient)
    data['will_last'] = (data['battery_duration'] > battery_capacity / 2).astype(int)

    # Define features (X) and target (y)
    X = data[['temperature', 'light_intensity', 'potentiometer_voltage', 'solar_panel_voltage']]
    y = data['will_last']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'energy_model.pkl')

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

def main():
    # Allow Serial to initialize
    time.sleep(2)
    print("Starting data collection...")

    # Set a counter for re-training frequency
    train_interval = 60  # Train the model every 60 seconds
    last_trained_time = time.time()

    while True:
        # Collect and save new data
        data = read_from_serial()
        if data:
            write_to_csv(data)
            print("Data saved:", data)
        
        # Check if it's time to re-train the model
        current_time = time.time()
        if current_time - last_trained_time >= train_interval:
            print("Re-training the model...")
            train_model()
            last_trained_time = current_time  # Reset the timer

        # Delay to match the Arduino's output frequency
        time.sleep(1)

if __name__ == "__main__":
    main()
# Final Product