import torch
import os
import cv2

def load_images_from_folder(folder): #Devuelve todas las imágenes de una carpeta
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#Crear la carpeta nueva
cropped_folder_path = './Cropped-IMGS/'
if not os.path.exists(cropped_folder_path):
    os.mkdir(cropped_folder_path)

for folder in os.listdir('./GTV-Database-UPC/'): #loop por todas las carpetas de imágenes
    folder_path = f'./GTV-Database-UPC/{folder}/'
    imgs = load_images_from_folder(folder_path) #obtener todas las imágenes de una carpeta
    os.mkdir(f'{cropped_folder_path}{folder}')
    print(f'Cropping folder {folder}', end='')
    cnt = 0

    for frame in imgs:

        profile = False
        flip_flag =  False
        detected_profile_face = 0
        detected_face = 0
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Hacemos esto porque el clasificador necesita imagenes en blanco y negro
        faces = face_cascade.detectMultiScale(gray_image, 1.0485258, 5) #1.04 es el scaling de la imagen con respecto a las imagenes con las que se entrenó el modelo, 12 es el minNeighbours (afecta a la calidad y al numero de detecciones)

        if faces is ():
            profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            gray_profile_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Hacemos esto porque el clasificador necesita imagenes en blanco y negro
            profile_faces = profile_face_cascade.detectMultiScale(gray_profile_image, 1.0485258, 3) #1.04 es el scaling de la imagen con respecto a las imagenes con las que se entrenó el modelo, 12 es el minNeighbours (afecta a la calidad y al numero de detecciones)
            
            if profile_faces is (): #No ha detectado un perfil izquierdo, así que la imagen podría estar del perfil derecho (Haar Cascade no reconoce perfiles derechos)
                flipped = cv2.flip(frame, 1) #Hay que flippear la imagen porque el haar_cascade solo está entrenado para perfiles izquierdos
                flip_flag = True
                gray_flipped_image = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
                profile_faces = profile_face_cascade.detectMultiScale(gray_flipped_image, 1.0485258, 3) #1.04 es el scaling de la imagen con respecto a las imagenes con las que se entrenó el modelo, 12 es el minNeighbours (afecta a la calidad y al numero de detecciones)

            for(x,y,w,h) in profile_faces: #detectMultiScale devuelve un rectángulo. x,y son las coordenadas de la esquina superior izquierda, (x+w),(y+h) hacen la esquina inferior derecha pq en open cv el eje y va en positivo hacia abajo
            #cv2.rectangle(frame,(x,y),(x+w,y+h), (255,102,34), 4) # el parámetro 4 representa el grosor de la línea del rectángulo
                if flip_flag:
                  detected_profile_face_flipped = flipped[y:y+h, x:x+w]
                  detected_profile_face = cv2.flip(detected_profile_face_flipped, 1)
                else: detected_profile_face = frame[y:y+h, x:x+w] #hace el crop de la cara detectada (el interior del rectángulo)
            profile = True
        
        else:
            for(x,y,w,h) in faces: #detectMultiScale devuelve un rectángulo. x,y son las coordenadas de la esquina superior izquierda, (x+w),(y+h) hacen la esquina inferior derecha pq en open cv el eje y va en positivo hacia abajo
            #cv2.rectangle(frame,(x,y),(x+w,y+h), (255,102,34), 4) # el parámetro 4 representa el grosor de la línea del rectángulo
                detected_face = frame[y:y+h, x:x+w] #hace el crop de la cara detectada (el interior del rectángulo)
            
        
        cnt += 1
        image_path = f'{cropped_folder_path}{folder}/{folder}_{cnt:03}.png'
        if profile:
            cv2.imwrite(image_path, detected_profile_face)
        else: cv2.imwrite(image_path, detected_face) 
        #cv2.imshow('face', detected_face)
        #k = cv2.waitKey(1) #Se declara una variable con el resultado de llamar a WaitKey porque mejora sustancialmente el tiempo de ejecución, si no, no se puede ejecutar el if del guardado tan rápido como quiera el usuario hacer click
        print('.', end='')
        
        

    print(f'completed')

if __name__ == "__main__":
    pass
