import os
import cv2

def load_images_from_folder(folder): # Return all the images on a folder
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
    

# Creates a new folder

cropped_folder_path = './Datasets/Cropped-IMGS-1-supervised/'
if not os.path.exists(cropped_folder_path):
    os.mkdir(cropped_folder_path)

def run():

    for folder in os.listdir('./Datasets/GTV-Database-UPC/'): # Loop through all the folders
        folder_path = f'./Datasets/GTV-Database-UPC/{folder}/'
        imgs = load_images_from_folder(folder_path) # Take all the images of a folder
        print(f'Cropping folder {folder}', end='')
        cnt = 0

        for frame in imgs:

            profile = False
            flip_flag =  False
            detected_profile_face = 0
            detected_face = 0
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Classifier needs Black and White images 
            faces = face_cascade.detectMultiScale(gray_image, 1.0485258, 5,  minSize=(70, 70)) # 1.04 is the scaling of the image with respect to the images with whic te model was trained, 12 is the minNeighbours (affects the quality and the number of detections)


            if faces is ():
                profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
                gray_profile_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Classifier needs Black and White images
                profile_faces = profile_face_cascade.detectMultiScale(gray_profile_image, 1.0485258, 3, minSize=(70, 70)) # 1.04 is the scaling of the image with respect to the images with whic te model was trained, 12 is the minNeighbours (affects the quality and the number of detections)
                
                if profile_faces is (): # No left profile has been detected, so the image could be on the right profile (Haar Cascade doesn't recognize right profiles)
                    flipped = cv2.flip(frame, 1) # The image has to be flipped due to Haar Cascade is not trained to recognize faces on right profiles
                    flip_flag = True
                    gray_flipped_image = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
                    profile_faces = profile_face_cascade.detectMultiScale(gray_flipped_image, 1.0485258, 3,  minSize=(70, 70)) 

                for(x,y,w,h) in profile_faces: # detectMultiScale returns a rectangle. x,y are the coordinates of the upper left corner, (x+w),(y+h) make the lower right corner pq in open cv the y-axis goes positive downwards
                #cv2.rectangle(frame,(x,y),(x+w,y+h), (255,102,34), 4) # the fourth parameter represents the wide of the line in the rectangle representation
                    if flip_flag:
                        detected_profile_face_flipped = flipped[y:y+h, x:x+w]
                        detected_profile_face = cv2.flip(detected_profile_face_flipped, 1)
                    else: 
                        detected_profile_face = frame[y:y+h, x:x+w] # Makes the cropping of the face detected (the inside of the rectangle)
                profile = True
            
            else:
                for(x,y,w,h) in faces:
                    detected_face = frame[y:y+h, x:x+w]
                
            
            cnt += 1
            image_path = f'{cropped_folder_path}{folder}_{cnt:03}.bmp'
            if profile:
                cv2.imwrite(image_path, detected_profile_face)
            else: 
                cv2.imwrite(image_path, detected_face) 
            print('.', end='')
            
        

        print(f'completed')

    delete_images_from_folder(cropped_folder_path,5)## This method deletes all the images of not detected faces
                                                    ## All these images are 4kb, that's why the threshold is set to 5

    print('Folder is now clean of empty detections')

if __name__ == "__main__":
    run()