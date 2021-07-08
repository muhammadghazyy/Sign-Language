import tensorflow as tf
import cv2
import numpy as np

letters_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'K',10:'L',11:'M',12:'N',13:'O',14:'P',15:'Q',16:'R',17:'S',18:'T',19:'U',20:'V',21:'W',22:'X', 23:'Y'}
model = tf.keras.models.load_model('sign_model.h5')
input_shape = (28,28,1)

capture = cv2.VideoCapture(1)

while True:
    ret,frame = capture.read()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    area = frame[100:300,350:550]
    area = cv2.cvtColor(area , cv2.COLOR_BGR2GRAY)
    area = cv2.resize(area , (28,28), interpolation = cv2.INTER_AREA)

    cv2.rectangle(frame, (300,100),(550,350), (255,255,255),2)
    area = area.reshape(1,28,28,1)

    pred = model.predict_classes(area,1,verbose=0)[0]
    cv2.putText(frame,letters_dict[pred], (300,85),font,1,(255,255,255),2)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
