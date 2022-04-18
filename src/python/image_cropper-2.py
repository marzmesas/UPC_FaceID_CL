import os
import cv2
import numpy as np

def load_images_from_folder(folder): #Devuelve todas las imágenes de una carpeta
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def delete_images_from_folder(folder, imgsz):
    if len(os.listdir(folder) ) != 0:
        for file in os.listdir(folder):
            if os.path.getsize(os.path.join(folder,file)) < imgsz * 1024:
                os.remove(os.path.join(folder,file))

def detect_face(frame, net):
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
        frame = cv2.resize(frame, (160,160), cv2.INTER_LINEAR)
    except Exception as e:
        pass
    return frame

#Crear la carpeta nueva

cropped_folder_path = './Datasets/Cropped-IMGS-2-supervised/'
if not os.path.exists(cropped_folder_path):
    os.mkdir(cropped_folder_path)

def run():

    #Load pretrained face detection model    
    net = cv2.dnn.readNetFromCaffe('./src/python/app/detection_model/deploy.prototxt.txt', './src/python/app/detection_model/res10_300x300_ssd_iter_140000.caffemodel')

    for folder in os.listdir('./Datasets/GTV-Database-UPC'): #loop por todas las carpetas de imágenes
        folder_path = f'./Datasets/GTV-Database-UPC/{folder}/'
        imgs = load_images_from_folder(folder_path) #obtener todas las imágenes de una carpeta
        ##########################################
        os.mkdir(f'{cropped_folder_path}{folder}')
        #########################################
        print(f'Cropping folder {folder}', end='')
        cnt = 0

        for frame in imgs:
            
            detected_face = detect_face(frame,net)
            cnt += 1
            if detected_face.shape[0]>0 and detected_face.shape[1]>0:
                detected_face = cv2.resize(detected_face, (160,160), interpolation=cv2.INTER_LINEAR)
                ##########################################################
                #image_path = f'{cropped_folder_path}{folder}_{cnt:03}.bmp'
                ###########################################################
                image_path = f'{cropped_folder_path}{folder}/{folder}_{cnt:03}.bmp'
                cv2.imwrite(image_path, detected_face) 
                cv2.imshow('face', detected_face)
                k = cv2.waitKey(1) #Se declara una variable con el resultado de llamar a Wait
                # Key porque mejora sustancialmente el tiempo de ejecución, si no, no se puede ejecutar el if del guardado tan rápido como quiera el usuario hacer click
                print('.', end='')
            
        print(f'completed')

    #delete_images_from_folder(cropped_folder_path,5)##Este método borra todas aquellas imágenes que se guardan sin detección de cara
                                                    ## Todas estas imágenes son de 4kb, por lo tanto ponemos el threshold en 5.
    #print('Folder is now clean of empty detections')

if __name__ == "__main__":
    run()