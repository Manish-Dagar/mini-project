import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

# Load the model
base_model = MobileNetV2(weights=None, include_top=False, pooling="avg", input_shape=(224, 224, 3))
head_model = Dense(1, activation="sigmoid")(base_model.output)
model = Model(inputs=base_model.input, outputs=head_model)
model.load_weights("ModelWeights.weights.h5")
print("Model loaded successfully with weights.")

# Function to preprocess frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return frame

# Start capturing from webcam
cap = cv2.VideoCapture(0)

frame_buffer = []
predictions = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Preprocess and store the frame
    processed_frame = preprocess_frame(frame)
    frame_buffer.append(processed_frame)

    # Keep buffer size fixed to 30 frames
    if len(frame_buffer) > 30:
        frame_buffer.pop(0)

    # Predict using the latest frame only
    input_frame = np.expand_dims(processed_frame, axis=0)  # shape: (1, 224, 224, 3)
    prediction = model.predict(input_frame, verbose=0)[0][0]
    label = "non-Violence" if prediction > 0.5 else "Violence"
    confidence = prediction if label == "Violence" else 1 - prediction
    predictions.append(label)

    # Display result
    cv2.putText(frame, f"{label} ({confidence:.2%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if label == 'Violence' else (0, 255, 0), 2)

    cv2.imshow('Live Violence Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
