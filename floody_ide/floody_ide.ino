
#include "EspDHT.h"
#include <WiFi.h>
#include "HX710B.h"
#include <WebServer.h>

const char* ssid = "Excitel_Wifi - 2.4G";
const char* password = "7380590909";

#define DHTPIN 13          // Pin for DHT sensor
#define DHTTYPE DHT11     // DHT 11
#define LM35PIN 27         // Pin for LM35 temperature sensor
#define FLOW_PIN 4        // Pin for water flow rate sensor
#define TRIG_PIN 10        // Pin for ultrasonic sensor trigger
#define ECHO_PIN 11        // Pin for ultrasonic sensor echo

const int DOUT = 21;   //pressure sensor data pin
const int SCLK  = 20;   //pressure sensor clock pin

float Humidity = 0.0;
float tem = 0.0;
float distance = 0.0;
float flowRate = 0.0;
float atmtem = 0.0;

HX710B pressure_sensor;
EspDHT dht;
WebServer server(80);

void web() {
    String webpage = "<!DOCTYPE html> <html>\n";
    webpage += "<head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0, user-scalable=no\">\n"; 
    webpage += "<title>RPi Pico W Sensors</title>";
    webpage += "<meta http-equiv=\"refresh\" content=\"1\">\n";
    webpage += "<h1>Sensor Data</h1>";
    webpage += "<p>Humidity: " + String(Humidity) + "%</p>";
    webpage += "<p>Atm TEmp: " + String(atmtem) + "C</p>";
    webpage += "<p>Temperature: " + String(tem) + "C</p>";
    webpage += "<p>Distance: " + String(distance) + "cm</p>";
    Serial.println(distance);
    webpage += "<p>Flow Rate: " + String(flowRate) + "L/min</p>";
    webpage += "<p>ATM Pressure: " + String(pressure_sensor.atm()) + "atm</p>";
    webpage += "</body></html>";
    
    server.send(200, "text/html", webpage);
}

void setup() {
    Serial.begin(115200);
    
    pressure_sensor.begin(DOUT, SCLK, 2);       // 2 gain channel
    dht.setup(DHTPIN, EspDHT::DHT11);
    pinMode(LM35PIN, INPUT);
    pinMode(FLOW_PIN, INPUT);
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.print("WiFi connected at IP Address ");
    Serial.println(WiFi.localIP());
    
    server.on("/", web);
    server.begin();
}

void loop() {
    server.handleClient();
    
    dht.readSensor();
    tem = (analogRead(LM35PIN) * 0.48828125) - 20;
    Humidity = dht.getHumidity();
    atmtem = dht.getTemperature();
    if (isnan(atmtem) || isnan(Humidity)) {
        Serial.println("Failed to read from DHT sensor!");
    }
    flowRate = flowdata();
    long durations, distance_cm;
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    durations = pulseIn(ECHO_PIN, HIGH);
    distance_cm = durations * 0.034 / 2;    // speed of sound in air - 0.034 cm/microsecond
    distance = distance_cm;
//    Serial.println(distance);
    if (pressure_sensor.is_ready()) {
        Serial.print("ATM: ");
        Serial.println(pressure_sensor.atm());
    }
    else {
        Serial.println("Pressure sensor not found.");
    }
}

float flowdata() {
    float TOTAL = 0;
    int pulseHigh = pulseIn(FLOW_PIN, HIGH);
    int pulseLow = pulseIn(FLOW_PIN, LOW);
    
    float pulseTime = pulseHigh + pulseLow;
    float frequency = 1000000.0 / pulseTime;
    Serial.println(frequency);
    float waterVolume = frequency / 7.5;
    float litersPerSecond = waterVolume / 60.0;
    
    if (frequency >= 0) {
        if (isinf(frequency)) {
            Serial.println("VOL.: 0.00");
            Serial.print("TOTAL: ");
            Serial.print(TOTAL);
            Serial.println(" L");
            return 0.00;
        }
        else {
            TOTAL += litersPerSecond;
            Serial.print("FREQUENCY: ");
            Serial.println(frequency);
            Serial.print("VOL.: ");
            Serial.print(waterVolume);
            Serial.println(" L/M");
            Serial.print("TOTAL: ");
            Serial.print(TOTAL);
            Serial.println(" L");
            return litersPerSecond;
        }
    }
    return 0.00;
}
