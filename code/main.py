# YOLOv5 and DJI Tello Integration with Python
# Requirements:
# - DJITelloPy: Python SDK for DJI Tello
# - OpenCV for video frame capture
# - PyTorch + YOLOv5 model (local or pretrained)

from djitellopy import Tello
import cv2
import torch
import time

# Load YOLOv5 model (ensure you have the correct path or pretrained model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # Confidence threshold

# Connect to Tello drone
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Start video stream
tello.streamon()
frame_read = tello.get_frame_read()

# Create OpenCV window
cv2.namedWindow("Tello YOLOv5", cv2.WINDOW_NORMAL)

try:
    while True:
        frame = frame_read.frame
        frame_resized = cv2.resize(frame, (640, 480))

        # Image Processing with YOLOv5
        results = model(frame_resized)
        results.render()  # Draw boxes on frame

        # Display the frame
        cv2.imshow("Tello YOLOv5", results.ims[0])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Keyboard Interrupt detected. Landing drone.")

finally:
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()