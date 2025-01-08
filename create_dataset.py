import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def preprocess_image(image):
    # Prepare feature vector
    data_aux = []

    # Convert image to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            # Normalize and append to data_aux
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

    # Pad to 84 features if only one hand is detected
    if len(data_aux) == 42:
        data_aux.extend([0] * 42)  # Padding with zeros for missing second hand

    return data_aux if len(data_aux) == 84 else None  # Ensure only complete vectors are used

# Example: Loading images and preparing data for Random Forest
dataset = []  # Store feature vectors
labels = []   # Store corresponding labels
DATA_DIR = 'dataset1'
# Replace with paths to your images and labels
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        image = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        feature_vector = preprocess_image(image)
        if feature_vector:  # Only add if 84 features are present
            dataset.append(feature_vector)
            labels.append(dir_)

# Convert dataset to numpy array and save using pickle
dataset = np.array(dataset)
labels = np.array(labels)
f = open('data.pickle', 'wb')
pickle.dump({'data': dataset, 'labels': labels}, f)
f.close()
