from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
sys.path.append('../')
import numpy as np
from evaluation import prediction
from model import base_model
import torch

global login,rec_frame, switch, capture, out, signup 
login=0
capture=0
signup=0
switch=1

# Make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./src/python/app/detection_model/deploy.prototxt.txt', './src/python/app/detection_model/res10_300x300_ssd_iter_140000.caffemodel')

# Instatiate flask app  
app = Flask(__name__, template_folder='./src/python/app/templates')

# Load unsupervised model
trained_modelq = base_model(pretrained=False)
trained_modelq.load_state_dict(torch.load(os.path.join('./src/python/saved_models', 'modelq.pt'),map_location=torch.device('cpu')))
# Load latents
latents = torch.load('./src/python/saved_models/latents.pt')

camera = cv2.VideoCapture(0)

def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
 
def gen_frames():  # Generate frame by frame from camera
    global out, login,rec_frame, capture, signup
    while True:
        success, frame = camera.read() 
        if success:                
            frame= detect_face(frame)   
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            if(login):
                login=0
                dist, idxs = prediction(frame, trained_modelq, latents, 1)
                if dist < 0.3:
                    print('You are logged')
                else:
                    print('Login denied')
            if(signup):
                pass
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Login':
            global login
            login=1
        elif  request.form.get('click') == 'Capture':
            global capture
            capture=not capture
            if(capture):
                time.sleep(4)   
        elif  request.form.get('click') == 'Sign up':
            global signup
            signup=1
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
                          
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()  