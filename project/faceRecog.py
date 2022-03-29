import cv2
import numpy as np 
import os        ## to read the files 


from sklearn.neighbors import KNeighborsClassifier


face_data = []          # to store the every imageof 1*10000 dimension in 2d matrix

labels=[]              # store the each image in 0,1,2 form which is y

dic = {
    0: 'amitabh',
    1: 'sharukh khan',
    2: 'Shrest'
}

idx = 0


for file in os.listdir("data"):
    data = np.load(f"./data/{file}")
    face_data.append(data)

    l = [idx for i in range(data.shape[0])]
    labels.extend(l)

    idx+=1




X = np.concatenate(face_data, axis=0)
Y = np.array(labels).reshape(-1, 1)


# make object of knn

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X, Y)



# ready for prediction by taking photo
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
    ret,frame =  cam.read()
    if ret == False:
         continue 

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#Chanfing the color of frame from color to gray to reduce heavier operations
    faces=model.detectMultiScale(gray_frame)#return the coordinates of multiple faces(if there) in a single frame
    if(len(faces) == 0):
        continue

    #lets take the other detections in the image are not face i.e false face detection and 
    # I want to detect i.e. take the face image with largest area,
    

    for face in faces:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)

        offset = 5
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        #resizing all obtained faces(cld be small ,big medium) to 100*100 size
        
        face_section = cv2.resize(face_section,(100,100))

        # query point converted from(100*100) to (1*10000)
        query = face_section.reshape(1, 10000)
        
        pred = knn.predict(query)[0]
        name = dic[int(pred)]
        
         
        cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        
    #for(x,y,w,h) in faces:
    #   cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)


    
    cv2.imshow("My video",frame)


    key_pressed = cv2.waitKey(1) & 0xFF  ## it is going to take the key from keyboard for quit
    # by tyhe help of bit wise conversion
    if key_pressed == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()


      
   