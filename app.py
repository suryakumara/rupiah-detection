import cv2
import math 
import pygame
from ultralytics import YOLO
import time

# Initialize Pygame mixer
pygame.mixer.init()


def play_audio(sound):
    # Play the audio file
    sound.play()

    # Wait until the audio is finished playing
    while pygame.mixer.get_busy():
        pygame.time.Clock().tick(0)

# Directory containing sound files
sound_dir = "sound"
# Nominal values and corresponding sound files
nominals = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
sound_files = {nominal: f"{sound_dir}/{nominal}.wav" for nominal in nominals}

# Load sound files
sounds = {nominal: pygame.mixer.Sound(file_path) for nominal, file_path in sound_files.items()}

sound_welcome = pygame.mixer.Sound("sound/welcome.wav") 
# Load YOLO model
model = YOLO("rupiah-detection.pt")

# initial_model.export(format="engine")

# model = YOLO("rupiah-detection.engine")
# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

play_audio(sound_welcome)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
          
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # Class name
            cls = int(box.cls[0])
            class_label = r.names[int(cls)]
            print('Detected:', class_label, "acc:", confidence)
    
            # Object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 2
            title = class_label + ", acc:" + str(confidence)
            # Put text on the image
            cv2.putText(img, title, org, font, fontScale, color, thickness)

            # Play sound if nominal is detected
            if confidence > 0.9:
                nominal_str = class_label.lower()  # Assuming class_label is like "1000"
                sound = sounds.get(nominal_str)
                if sound:
                    time.sleep(5)
                    play_audio(sound)
                    


    # Show the image
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
