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
os.mkdir(cropped_folder_path)

for folder in os.listdir('./GTV-Database-UPC/'): #loop por todas las carpetas de imágenes
    folder_path = f'./GTV-Database-UPC/{folder}/'
    imgs = load_images_from_folder(folder_path) #obtener todas las imágenes de una carpeta
    os.mkdir(f'{cropped_folder_path}{folder}')
    print(f'Cropping folder {folder}', end='')
    cnt = 0

    for frame in imgs:

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Hacemos esto porque el clasificador necesita imagenes en blanco y negro
        faces = face_cascade.detectMultiScale(gray_image, 1.0485258, 12) #1.04 es el scaling de la imagen con respecto a las imagenes con las que se entrenó el modelo, 12 es el minNeighbours (afecta a la calidad y al numero de detecciones)


        for(x,y,w,h) in faces: #detectMultiScale devuelve un rectángulo. x,y son las coordenadas de la esquina superior izquierda, (x+w),(y+h) hacen la esquina inferior derecha pq en open cv el eje y va en positivo hacia abajo
            #cv2.rectangle(frame,(x,y),(x+w,y+h), (255,102,34), 4) # el parámetro 4 representa el grosor de la línea del rectángulo
            detected_face = frame[y:y+h, x:x+w] #hace el crop de la cara detectada (el interior del rectángulo)
            
        
        cnt += 1
        image_path = f'{cropped_folder_path}{folder}/{folder}_{cnt:03}.png'
        cv2.imwrite(image_path, detected_face)
        #cv2.imshow('face', detected_face)
        #k = cv2.waitKey(1) #Se declara una variable con el resultado de llamar a WaitKey porque mejora sustancialmente el tiempo de ejecución, si no, no se puede ejecutar el if del guardado tan rápido como quiera el usuario hacer click
        print('.', end='')

    print(f'completed')

if __name__ == "__main__":
    pass
