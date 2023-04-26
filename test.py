import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import datetime
import smtplib
from flask import Flask, render_template, Response

app = Flask(__name__)

mixer.init() #initializing mixer 
sound = mixer.Sound('alarm.wav')  #alarm.wav file for ringing alarm

#These xml files are downloaded from google and used for detecting face , left eye and right eye
face = cv2.CascadeClassifier("haar cascade files\haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier("haar cascade files\haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier("haar cascade files\haarcascade_righteye_2splits.xml")

lbl = ['Close', 'Open']

model = load_model('models/trained.h5') #loading our model
path = os.getcwd() #It will give the path of current working directory
cap = cv2.VideoCapture(0) #initializing video object
font = cv2.FONT_HERSHEY_COMPLEX_SMALL #Font style for putting text on frame

def gen_frames():
    count = 0
    score = 0 #if score goes greater than 15 then ring the alarm
    thicc = 2 #border width
    rpred = [99]  #for taking class prediction of right eye
    lpred = [99]  #for taking class prediction of left eye

    while True:

        ret, frame = cap.read() #Capturing frame by frame
        height, width = frame.shape[:2]  #height and width of frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting frame to gray scale

        #Now we detect face , left_eye and right_eye 
        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)
        #drawing rectangle of frame size
        cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
       
        #Drawing rectangle around the face where x and y are coordinates of upper left corner, w-width, h-height
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,255,100) , 1 )
        cv2.rectangle(frame, (3, 3), (width-3, 35), (0, 0, 0), thickness=cv2.FILLED)


        for (x,y,w,h) in right_eye:
            r_eye = frame[y:y+h,x:x+w] #Extracting right eye from frame
            count= count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY) #Converting to gray scale
            r_eye = cv2.resize(r_eye,(24,24)) #Resizing image to (24,24) which input size for our model
            r_eye = r_eye/255
            r_eye = r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            predict_r = model.predict(r_eye) 
            rpred = np.argmax(predict_r,axis=1) #It will predict the class of the image means either eye is open or closed
            
            if rpred[0]==1: #If it gives 1 then eye is open because 1 is assigned to open
                lbl = 'Open' #Set label to open
            if rpred[0]==0: #If it gives 0 then eye is closed bacause 0 is assigned to closed
                lbl = 'Closed' #Set label to closed
                break


        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w] #Extracting left eye from frame
            count = count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY) #Converting to gray scale
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            predict_l = model.predict(l_eye) 
            lpred = np.argmax(predict_l,axis=1) #It will predict the class of the image means either eye is open or closed
            
            if(lpred[0]==1):  #If it gives 1 then eye is open because 1 is assigned to open
                lbl = 'Open' #Set label to open
            if(lpred[0]==0): #If it gives 0 then eye is closed bacause 0 is assigned to closed
                lbl = 'Closed' #Set label to closed
            break


        if(rpred[0]==0 and lpred[0]==0): #If both eyes are closed increment the score
            score = score+1
            #It will put the text closed in frame, we have given coordinated of upper left corner of block,font type,color,line type
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        else: #If eyes are not closed decrement the score
            score = score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


        if(score<0): #Restricting our score not to go less than zero
            score=0
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>15):
            #Person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'),frame) #Saving image to specified path


            try:
                sound.play()

            except:  # isplaying = False
                pass
            if(thicc<16): #This is for displaying motion of borders when alarm beeps
                thicc= thicc+2


            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) #Drawing rectangle
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): #If q is pressed break the loop
                break

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary = frame')

if __name__ == '__main__':
    app.run(debug=True)