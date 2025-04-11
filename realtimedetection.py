import cv2
from keras.models import model_from_json, Sequential
import numpy as np

# Load the model
def load_model():
    try:
        with open("facialemotionmodel.json", "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={"Sequential": Sequential})
        model.load_weights("facialemotionmodel.h5")
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# Load Haar Cascade
def load_haar_cascade():
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    if face_cascade.empty():
        print("Error: Haar Cascade file not found.")
        exit()
    return face_cascade

# Feature extraction function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Main function for real-time detection
def main():
    model = load_model()
    face_cascade = load_haar_cascade()

    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    # Open webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    print("Press 'q' to exit.")
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            try:
                face_resized = cv2.resize(face, (48, 48))
                features = extract_features(face_resized)
                prediction = model.predict(features)
                label = labels[prediction.argmax()]
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()