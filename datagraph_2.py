import urllib.request
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re
from datetime import datetime

# Initialize lists to store sensor data
humidity_data = []
temperature_data = []
flow_rate_data = []
distance_data = []
time_data = []

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Function to fetch data from the server
def fetch_data():
    url = 'http://192.168.99.44/'  # Replace with your server IP address or domain name
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8').splitlines()
        html_response = str(data)
        # Extract individual sensor data using regular expressions
        humidity_match = re.search(r'Humidity: (\d+\.\d+)%', html_response)
        temperature_match = re.search(r'Temperature: (\d+\.\d+)C', html_response)
        distance_match = re.search(r'Distance: (\d+\.\d+)cm', html_response)
        flow_rate_match = re.search(r'Flow Rate: (\d+\.\d+)L/min', html_response)
        pressure_match = re.search(r'ATM Pressure: ([\d.-]+)atm', html_response)

        # Check if all matches are found
        if humidity_match and temperature_match and distance_match and flow_rate_match and pressure_match:
            # Extract values from matches
            humidity = float(humidity_match.group(1))
            temperature = float(temperature_match.group(1))
            distance = float(distance_match.group(1))
            flow_rate = float(flow_rate_match.group(1))

            humidity_data.append(humidity)
            temperature_data.append(temperature)
            distance_data.append(distance)
            flow_rate_data.append(flow_rate)
            time_data.append(datetime.now())

# Function to update the plot
def update_plot(frame):
    fetch_data()
    ax.clear()
    ax.plot(time_data, humidity_data, label='Humidity (%)')
    ax.plot(time_data, temperature_data, label='Temperature (°C)')
    ax.plot(time_data, flow_rate_data, label='Flow Rate (L/min)')
    ax.plot(time_data, distance_data, label='Distance (cm)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Sensor Data Over Time')
    ax.legend()
    ax.grid(True)

# Animate the plot

ani = FuncAnimation(fig, update_plot, interval=500)  # Update every 2 seconds
plt.show()
