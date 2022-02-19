import torch
import os
import cv2

frame = cv2.imread('D:/Formacion_Academica/Posgrado_UPC/Projects/UPC_FaceID_CL/GTV-Database-UPC/ID01/ID01_001.bmp', cv2.IMREAD_UNCHANGED)

count = 0
count_neg = 0


while True:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Hacemos esto porque el clasificador necesita imagenes en blanco y negro
    faces = face_cascade.detectMultiScale(gray_image, 1.0485258, 12) #1.04 es el scaling de la imagen con respecto a las imagenes con las que se entrenó el modelo, 12 es el minNeighbours (afecta a la calidad y al numero de detecciones)
    print(faces)

    for(x,y,w,h) in faces: #detectMultiScale devuelve un rectángulo. x,y son las coordenadas de la esquina superior izquierda, (x+w),(y+h) hacen la esquina inferior derecha pq en open cv el eje y va en positivo hacia abajo
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,102,34), 4) # el parámetro 4 representa el grosor de la línea del rectángulo

        
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) #Se declara una variable con el resultado de llamar a WaitKey porque mejora sustancialmente el tiempo de ejecución, si no, no se puede ejecutar el if del guardado tan rápido como quiera el usuario hacer click
    if k == ord('q'):
        break
    
    
frame.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    pass
