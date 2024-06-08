
import asyncio
import re

from telegram import Bot
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import time

botToken ='7036389404:AAHtsOyXpX-ZrXGbcV1pCpjNwGbSABCTAxg'
bot = Bot(token = botToken)
channel_id ='1135126159'
# Initialize lists to store sensor data
temperature_data = []
pressure_data = []
altitude_data = []
dht_temperature_data = []
humidity_data = []
distance_data = []
water_temperature_data = []
flow_rate_data = []
total_liters_data = []
latitude_data = []
longitude_data = []
time_data = []
# dis=0.0
# flr=0.0
c=0
start_red = time.time()

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))
location_map='Alert! Flood parameters above threshold./n AT: http://maps.google.com/?q='

async def send_message(text,chat_id):

    async with bot:
        await bot.send_message(text=text,chat_id=chat_id)

# Function to fetch data from the server
def fetch_data():
    # global st_r
    url = 'http://192.168.115.35:5000/'  # Replace with your server IP address or domain name
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8').splitlines()
        html_response = data
        print(html_response)

        latitude = None
        longitude = None

        # Regular expressions to match latitude and longitude patterns
        latitude_pattern = r'Latitude: ([\d.]+)'
        longitude_pattern = r'Longitude: ([\d.]+)'

        # Iterate through the HTML response to find latitude and longitude data
        for line in html_response:
            latitude_match = re.search(latitude_pattern, line)
            if latitude_match:
                latitude = float(latitude_match.group(1))

            longitude_match = re.search(longitude_pattern, line)
            if longitude_match:
                longitude = float(longitude_match.group(1))
            # print("Latitude:", latitude)
            # print("Longitude:", longitude)
            location_map = 'Alert! Flood parameters above threshold./n AT: http://maps.google.com/?q='
            location_map = location_map + str(latitude) +',' + str(longitude)
    # html_response = ['<!DOCTYPE html>', '<html lang="en">', '<head>', '    <meta charset="UTF-8">',
    #                  '    <meta http-equiv="X-UA-Compatible" content="IE=edge">',
    #                  '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
    #                  '    <title>Sensor Data</title>', '</head>', '<body>', '    <h1>Sensor Data</h1>',
    #                  '    <p>Temperature: 31.4</p>', '    <p>Pressure: 91257</p>', '    <p>Altitude: 874.1</p>',
    #                  '    <p>DHT_T: 30</p>', '    <p>Humidity: 20</p>', '    <p>Distance: 17.73</p>',
    #                  '    <p>Water_T: 29.125</p>', '    <p>Flow Rate: 0.20 l/min</p>',
    #                  '    <p>Total Liters: 0.0</p>', '    <p>Latitude: 12.908207</p>',
    #                  '    <p>Longitude: 77.566792</p>', '', ' ', '</body>', '</html>', '                                                        ']

    for line in html_response:
        # dis = 0.0
        # flr =o.o
        line = line.strip().strip('<p>').strip('</')
        # print(line)
        end_red = time.time()
        if "Temperature:" in line:
            temperature_data.append(float(line.split(":")[-1]))
        elif "Pressure:" in line:
            pressure_data.append((float(line.split(":")[-1]))/1000)
        elif "Altitude:" in line:
            altitude_data.append((float(line.split(":")[-1]))/10)

        elif "DHT_T:" in line:
            print("here")
            dht_temperature_data.append(float(line.split(":")[-1]))
            print(dht_temperature_data[0])
        elif "Humidity:" in line:
            humidity_data.append(float(line.split(":")[-1]))
        elif "Distance:" in line:
            dis = float(line.split(":")[-1])
            print("time=")
            print((time.time() - start_red)%10)
            if dis < 16 and (time.time() - start_red)%10 >= 5:
                print("send")
                # asyncio.run(bot.send_message(chat_id=channel_id,text=location_map))
                asyncio.run(bot.sendMessage(chat_id=channel_id, text=location_map))
            distance_data.append(dis)
        elif "Water_T:" in line:
            water_temperature_data.append(float(line.split(":")[-1]))
        elif "Flow Rate:" in line:
            flr=float(line.split(":")[-1][:-6])*5
            flow_rate_data.append(flr)
        elif "Total Liters:" in line:
            total_liters_data.append(float(line.split(":")[-1]))
        elif "Latitude:" in line:
            latitude_data.append(float(line.split(":")[-1]))
        elif "Longitude:" in line:
            longitude_data.append(float(line.split(":")[-1]))
        # if dis < 16 and flr < 50:
        #     asyncio.run(bot.send_message(chat_id=channel_id, text=location_map))
    print(temperature_data[0])
    print(pressure_data[0])
    print(altitude_data[0])
    print(dht_temperature_data[0])
    print(humidity_data[0])
    print(distance_data[0])
    print(water_temperature_data[0])

    # Append current time
    # time_data.append(datetime.now())
    sec = time.time()
    time_data.append(time.ctime(sec)[-7:-5])
    print(time.ctime(sec)[-13:-5])


# Function to update the plot
def update_plot(frame):
    fetch_data()
    ax.clear()
    ax.plot( temperature_data, label='Temperature (°C)')
    ax.plot(time_data, pressure_data, label='Pressure')
    # ax.plot(time_data, altitude_data, label='Altitude')
    # ax.plot(time_data, dht_temperature_data, label='DHT Temperature')
    ax.plot(time_data, humidity_data, label='Humidity (%)')
    ax.plot(time_data, distance_data, label='Distance (cm)')
    ax.plot(time_data, water_temperature_data, label='Water Temperature (°C)')
    ax.plot(time_data, flow_rate_data, label='Flow Rate (L/min)')
    # ax.plot(time_data, total_liters_data, label='Total Liters')
    # ax.plot(time_data, latitude_data, label='Latitude')
    # ax.plot(time_data, longitude_data, label='Longitude')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Sensor Data Over Time')
    ax.legend()
    ax.grid(True)


# Animate the plot
ani = FuncAnimation(fig, update_plot, interval=1000)  # Update every 2 seconds
c=c+1
if c == 10:
    start_red=time.time()
    c=0
plt.show()
