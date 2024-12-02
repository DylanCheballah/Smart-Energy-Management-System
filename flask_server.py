from flask import Flask, jsonify
from flask_cors import CORS
import csv
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to the shared CSV file
csv_file_path = 'sensor_data.csv'

@app.route('/data', methods=['GET'])
def get_data():
    """Serve the latest data from the shared CSV file."""
    if not os.path.exists(csv_file_path):
        return jsonify({"error": "No data available"}), 404

    try:
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            if rows:
                latest_row = rows[-1]  # Get the most recent data
                return jsonify({
                    "temperature": float(latest_row["temperature"]),
                    "light_intensity": float(latest_row["light_intensity"]),
                    "potentiometer_voltage": float(latest_row["potentiometer_voltage"]),
                    "solar_panel_voltage": float(latest_row["solar_panel_voltage"]),
                    "timestamp": latest_row.get("timestamp", "N/A")  # Optional timestamp field
                })
            else:
                return jsonify({"error": "No data available"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run the server on all available addresses (host="0.0.0.0")
    app.run(debug=True, host="0.0.0.0", port=5000)
