import cv2.data
import numpy as np
import cv2
from tf_keras.models import load_model

# loading training models:
age_model = load_model('./Models/Age_model.h5')
gen_model = load_model('./Models/Gender_model.h5')

# load face detector:
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + './Models/haarcascade_frontalface_default (2).xml')

# start webcam ad default:
v_cap = cv2.VideoCapture(0)


# while loop for main program function:
while True:
    ret, frame = v_cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (64, 64))
        face_img = np.expand_dims(face_img, axis=0) / 255.0

        # Predict gender and age
        gender = gen_model.predict(face_img)
        age = age_model.predict(face_img)

        gender_label = 'Male' if gender[0][0] > 0.5 else 'Female'
        age_label = int(age[0][0])

        # Display results
        label = f'{gender_label}, {age_label} yrs'
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Real-Time Gender and Age Predictor', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

v_cap.release()
cv2.destroyAllWindows()




