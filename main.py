import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps


model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()


cap = cv2.VideoCapture(0)


scale_factor = 1.5


cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detection', 800, 600)  

while True:
    ret, frame = cap.read()


    frame = cv2.resize(frame, (224, 224))
    frame_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_array = np.expand_dims(frame_array, axis=0)


    normalized_frame_array = (frame_array.astype(np.float32) / 127.5) - 1


    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_frame_array


    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]


    h, w, _ = frame.shape
    box_color = (0, 255, 0)  
    box_thickness = 2

    font_scale = 0.5  
    cv2.putText(frame, f'{class_name[2:]}: {confidence_score:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 1, cv2.LINE_AA)


    box_x = int(w / 2 - w / (2 * scale_factor))
    box_y = int(h / 2 - h / (2 * scale_factor))
    box_w = int(w / scale_factor)
    box_h = int(h / scale_factor)

    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, box_thickness)


    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
