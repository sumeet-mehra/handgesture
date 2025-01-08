import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=2)

# Define labels dictionary
labels_dict = {0: 'Rest', 1: 'Sit', 2: 'Put', 3: 'Stand', 4: 'Eat', 5: 'Drink', 6: 'Cut', 7: 'Push', 8: 'Wash', 9: 'Drop', 10: 'Hit', 11: 'Stop', 12: 'Clean', 13: 'Pour', 14: 'Sorry', 15: 'Alone', 16: 'Angry', 17: 'Sick', 18: 'House', 19: 'Everyday', 20: 'Money', 21: 'Animal', 22: 'Car', 23: 'Wish', 24: 'Problem'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

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

    cv2.imshow('Hand Sign Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
