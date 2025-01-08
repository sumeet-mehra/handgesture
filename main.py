import pickle
import cv2
import mediapipe as mp
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout

class HandSignDetectionApp(App):
    def build(self):
        # Initialize the model
        model_dict = pickle.load(open('model.p', 'rb'))
        self.model = model_dict['model']

        # Initialize Mediapipe hands solution
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=True, 
            min_detection_confidence=0.5, 
            max_num_hands=2
        )

        # Define labels dictionary
        self.labels_dict = {
            0: 'Rest', 1: 'Sit', 2: 'Put', 3: 'Stand', 4: 'Eat', 5: 'Drink', 6: 'Cut', 7: 'Push', 8: 'Wash',
            9: 'Drop', 10: 'Hit', 11: 'Stop', 12: 'Clean', 13: 'Pour', 14: 'Sorry', 15: 'Alone', 16: 'Angry',
            17: 'Sick', 18: 'House', 19: 'Everyday', 20: 'Money', 21: 'Animal', 22: 'Car', 23: 'Wish', 24: 'Problem'
        }

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Create the UI layout
        layout = BoxLayout(orientation='vertical')

        # Kivy Image widget for displaying video
        self.video_display = Image()
        layout.add_widget(self.video_display)

        # Label for displaying predictions
        self.prediction_label = Label(text="Prediction: None", size_hint=(1, 0.1), font_size='20sp')
        layout.add_widget(self.prediction_label)

        # Schedule the update function
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        data_aux = []
        x_ = []
        y_ = []

        # Convert the frame to RGB for Mediapipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    self.mp_hands.HAND_CONNECTIONS,  # hand connections
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

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
            prediction = self.model.predict(data_aux)
            predicted_label = self.labels_dict[int(prediction[0])]

            # Update the prediction label
            self.prediction_label.text = f"Prediction: {predicted_label}"

        # Convert the frame to a Kivy texture
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.video_display.texture = texture

    def on_stop(self):
        # Release the video capture when the app is closed
        self.cap.release()

if __name__ == '__main__':
    HandSignDetectionApp().run()
