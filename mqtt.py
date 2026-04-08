import base64
import cv2
import numpy as np
import os
import paho.mqtt.client as mqtt

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"  # Replace with your broker
MQTT_PORT = 1883
TOPICS = [("scd/images", 0), ("camera/images", 0)]

# Image counters
image_counters = {
    "scd/images": 0,
    "camera/images": 0
}

# Define custom save locations for each topic
SAVE_PATHS = {
    "scd/images": r"runs/folder_a",
    "camera/images": r"runs/folder_b"
}

# Make sure folders exist
for path in SAVE_PATHS.values():
    os.makedirs(path, exist_ok=True)

# Callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker!")
        client.subscribe(TOPICS)
    else:
        print("Failed to connect, return code:", rc)

def on_message(client, userdata, msg):
    try:
        image_counters[msg.topic] += 1
        index = image_counters[msg.topic]

        print(f"\nReceived message from topic: {msg.topic} | Size: {len(msg.payload)} bytes")
        print(f"Payload preview (first 50 chars): {msg.payload[:50]}")

        # Decode base64 to bytes
        try:
            img_bytes = base64.b64decode(msg.payload)
        except Exception as e:
            print(f"Base64 decoding failed for topic {msg.topic}: {e}")
            return

        # Convert bytes to numpy array and decode image
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is not None:
            # Save image to the defined folder
            folder_path = SAVE_PATHS[msg.topic]
            file_path = os.path.join(folder_path, f"img_{index:05d}.png")
            cv2.imwrite(file_path, img)
            print(f"Saved image to {file_path}")
        else:
            print(f"cv2 failed to decode image from topic {msg.topic}. Possibly not a valid image.")
    except Exception as e:
        print("Error processing message:", e)

# MQTT Client setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()