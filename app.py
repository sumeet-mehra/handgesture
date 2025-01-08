import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Hand Sign Detection",
    page_icon="👋",
    layout="wide"
)

# Add title and description
st.title("Real-time Hand Sign Detection")
st.markdown("""
This application detects and interprets hand signs in real-time using your webcam.
The model can recognize 25 different signs including common actions and words.
""")

# Load the trained model
@st.cache_resource
def load_model():
    model_dict = pickle.load(open('model.p', 'rb'))
    return model_dict['model']

model = load_model()

# Initialize Mediapipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=2)

# Define labels dictionary
labels_dict = {
    0: 'Rest', 1: 'Sit', 2: 'Put', 3: 'Stand', 4: 'Eat', 
    5: 'Drink', 6: 'Cut', 7: 'Push', 8: 'Wash', 9: 'Drop', 
    10: 'Hit', 11: 'Stop', 12: 'Clean', 13: 'Pour', 14: 'Sorry', 
    15: 'Alone', 16: 'Angry', 17: 'Sick', 18: 'House', 19: 'Everyday',
    20: 'Money', 21: 'Animal', 22: 'Car', 23: 'Wish', 24: 'Problem'
}

# Create two columns
col1, col2 = st.columns(2)

# Add legend to the second column
with col2:
    st.subheader("Available Signs")
    chunks = np.array_split(list(labels_dict.items()), 5)
    for chunk in chunks:
        cols = st.columns(len(chunk))
        for i, (key, value) in enumerate(chunk):
            with cols[i]:
                st.button(value, key=f"sign_{key}", disabled=True)

# Add webcam feed to the first column
with col1:
    st.subheader("Webcam Feed")
    # Start the webcam
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button = st.button("Stop")

    while cap.isOpened() and not stop_button:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                    
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Ensure the data_aux has the correct number of features
            if len(data_aux) == 42:
                # Duplicate the features to match the expected 84 features
                data_aux = data_aux * 2

            # Convert to numpy array and reshape for the model
            data_aux = np.array(data_aux).reshape(1, -1)

            # Predict the label
            prediction = model.predict(data_aux)
            predicted_label = labels_dict[int(prediction[0])]

            # Display the predicted label on the frame
            cv2.putText(frame, predicted_label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        # if results.multi_hand_landmarks:
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         mp_drawing.draw_landmarks(
        #             frame,
        #             hand_landmarks,
        #             mp_hands.HAND_CONNECTIONS,
        #             mp_drawing_styles.get_default_hand_landmarks_style(),
        #             mp_drawing_styles.get_default_hand_connections_style()
        #         )
                
        #         for i in range(len(hand_landmarks.landmark)):
        #             x = hand_landmarks.landmark[i].x
        #             y = hand_landmarks.landmark[i].y
        #             x_.append(x)
        #             y_.append(y)

        #         for i in range(len(hand_landmarks.landmark)):
        #             x = hand_landmarks.landmark[i].x
        #             y = hand_landmarks.landmark[i].y
        #             data_aux.append(x - min(x_))
        #             data_aux.append(y - min(y_))

        #     if len(data_aux) == 42:
        #         data_aux = data_aux * 2

        #     data_aux = np.array(data_aux).reshape(1, -1)
        #     prediction = model.predict(data_aux)
        #     predicted_label = labels_dict[int(prediction[0])]

        #     # Display prediction with a more visible design
        #     cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
        #     cv2.putText(
        #         frame, 
        #         f"Sign: {predicted_label}", 
        #         (20, 60), 
        #         cv2.FONT_HERSHEY_SIMPLEX, 
        #         2, 
        #         (255, 255, 255), 
        #         3, 
        #         cv2.LINE_AA
        #     )

        # Convert frame to PIL Image and display it
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

    cap.release()

# Add footer with instructions
st.markdown("""
---
### Instructions:
1. Allow camera access when prompted
2. Show your hand signs clearly in the camera view
3. The detected sign will be displayed above the video feed
4. Click 'Stop' to end the session

**Note:** Make sure you have good lighting and a clear background for better detection.
""")