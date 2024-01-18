
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

# Capture live video stream
cap = cv2.VideoCapture(0)

# Serial communication setup
ser = serial.Serial('COM6', 9600)  # Change 'COM3' to your Arduino port

while True:
    # Capture frame-by-frame