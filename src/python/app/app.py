from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
sys.path.append('..')
import numpy as np
from evaluation import prediction, compute_embeddings
from model import base_model
import torch
from flask import*

global login,rec_frame, switch, capture, out, signup

login=0
capture=0
signup=0
switch=1
log_correct=False


#Load dataset path
config_fixed_app={}
config_app={}
config_app["batch_size"]=16
dataset_path = "../../../Datasets/Cropped-IMGS-2-supervised-train"
logs_path = "./logs"
config_fixed_app["image_path"] = dataset_path
config_fixed_app["logs_path"] = logs_path

# Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./detection_model/deploy.prototxt.txt', './detection_model/res10_300x300_ssd_iter_140000.caffemodel')

# Instatiate flask app  
app = Flask(__name__, template_folder='./templates')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


# Load supervised model
trained_modelq = base_model(pretrained=False)
trained_modelq.load_state_dict(torch.load('../saved_models/model_Contrastive.pt',map_location=torch.device('cpu')))
camera = cv2.VideoCapture(0)

latents, labels, _ , _ , _ = compute_embeddings(modelq=trained_modelq,config=config_app,config_fixed=config_fixed_app,testing=True,show_latents=True)


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

def take_picture(Persona,count, folder_path):
    _,frame = camera.read()
    frame = detect_face(frame)
    cv2.imwrite(folder_path+'/ID{Persona}_{N}.bmp'.format(Persona="%02d" % (Persona,),N="%02d" % (count+1,)),frame)
    print('Saved image: ','ID_{Persona}_{N}.bmp'.format(Persona=Persona,N=count))
 
def gen_frames():  # Generate frame by frame from camera
    global out, login,rec_frame, capture, signup,latents,labels,log_correct
    while True:
        success, frame = camera.read() 
        if success:                
            frame= detect_face(frame)   
            if(capture):
                capture=0
            if(login):
                #Function to check if you can log in or not (are you in the database?)
                log_correct=False
                dist,labels_predichas=prediction(image=frame,modelq=trained_modelq,latents=latents,topk=3,labels=labels)
                dist = round(dist,2)
                print(dist)
                if dist < 4.5:
                    log_correct=True
                    print('You are logged')
                else:
                    print('Login denied',"error")
                login=0

            if(signup):
                
                folder_count = 0  # type: int
                for folders in os.listdir(config_fixed_app["image_path"]):
                    folder_count += 1  # increment counter
                print("There are {0} folders".format(folder_count))
                new_username = f'ID{folder_count+1}'
                new_folder = (dataset_path+f"/{new_username}")
                os.mkdir(new_folder)
                pictures = 0
                while pictures <5:
                    _,frame = camera.read()
                    frame = detect_face(frame)
                    take_picture(folder_count+1,pictures, new_folder)

                    try:
                        ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        pass
                    pictures+=1
                    time.sleep(1)
                signup=0
                latents, labels, _ , _ , _ = compute_embeddings(modelq=trained_modelq,config=config_app,config_fixed=config_fixed_app,testing=True,show_latents=False)

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
            time.sleep(2)
            if log_correct:
                flash("Succesfully logged in")
                return redirect(url_for('index'))
            else:
                flash("Login denied")
                return redirect(url_for('index')) 
        elif  request.form.get('click') == 'Sign up':
            login=1
            time.sleep(2)
            if log_correct:
                flash("You are already registered!")
                return redirect(url_for('index'))
            else:
                global signup
                signup=1
                time.sleep(10)
                flash("Sign up completed")
                return redirect(url_for('index'))
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