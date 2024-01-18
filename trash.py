import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# Load TensorFlow Lite model and label mapping
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_tensor_index = interpreter.get_input_details()[0]['index']
output_tensor_index = interpreter.get_output_details()[0]['index']

with open('labels.txt', 'r') as file:
    labels = [line.strip() for line in file]

desired_height = 224
desired_width = 224

def preprocess_frame(frame):
    # Resize the frame to match the input size expected by the model
    input_shape = (desired_height, desired_width)
    resized_frame = cv2.resize(frame, input_shape)

    # Normalize the pixel values to be in the range [0, 1]
    input_data = resized_frame.astype(np.float32) / 255.0

    # Expand dimensions to create a batch of size 1
    input_data = np.expand_dims(input_data, axis=0)

    return input_data

def postprocess_output(output_data, labels):
    # Assuming a classification task with softmax activation
    probabilities = tf.nn.softmax(output_data[0])

    # Get the index with the highest probability
    predicted_class = np.argmax(probabilities)

    # Get the corresponding label
    result_label = labels[predicted_class]

    # Return the result in a format you desire
    results = {'label': result_label, 'probability': probabilities[predicted_class]}

    return results

def visualize_results(frame, results):
    # Extract label and probability from the results
    label = results['label']
    probability = results['probability']

    # Display the label and probability on the frame
    text = f"{label}: {probability:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

# Capture live video stream
cap = cv2.VideoCapture(0)

# Serial communication setup
ser = serial.Serial('COM6', 9600)  # Change 'COM3' to your Arduino port

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame for the model input
    input_data = preprocess_frame(frame)

    # Run inference
    interpreter.set_tensor(input_tensor_index, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_tensor_index)

    # Post-process and visualize results
    results = postprocess_output(output_data, labels)
    frame_with_results = visualize_results(frame, results)

    # Serial communication to Arduino based on detected label

    if results['label'] == 'plastic':
      ser.write(b'1') # Send 1 to move servo to 180 degrees
      time.sleep(.5)
      ser.write(b'0') # Send 0 to return to 90 degrees

    if results['label'] == 'paper':
      ser.write(b'2') # Send 2 to move servo to 0 degrees 
      time.sleep(.5)
      ser.write(b'0') # Send 0 to return to 90 degrees
    if results['label'] == 'glass':
      ser.write(b'1') # Send 1 to move servo to 180 degrees
      time.sleep(.5)
      ser.write(b'0') # Send 0 to return to 90 degrees
    if results['label'] == 'metal':
      ser.write(b'1') # Send 1 to move servo to 180 degrees
      time.sleep(.5)
      ser.write(b'0') # Send 0 to return to 90 degree
    if results['label'] == 'trash':
      ser.write(b'1') # Send 1 to move servo to 180 degrees
      time.sleep(.5)
      ser.write(b'0') # Send 0 to return to 90 degrees
    if results['label'] == 'compost':
      ser.write(b'2') # Send 1 to move servo to 180 degrees
      time.sleep(.5)
      ser.write(b'0') # Send 0 to return to 90 degrees
    # Display the resulting frame
    cv2.imshow('Object Detection', frame_with_results)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close serial connection
cap.release()
cv2.destroyAllWindows()
ser.close()
