import cv2
import numpy as np 

cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_data = []
skip = 0

name=input("Enter your name")

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
    sorted(faces, key = lambda f:f[2]*f[3])

    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)

        offset = 5
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        #resizing all obtained faces(cld be small ,big medium) to 100*100 size
        face_section = cv2.resize(face_section,(100,100))
        
        #takes 24frames in 1 sec
        # put a logic of appending every 10thframe and not all the frames 

        skip += 1
        #giving some pause while taking the frame 
        if skip%10 ==0:
            face_data.append(face_section)
            print(len(face_data))   



    #for(x,y,w,h) in faces:
    #   cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)


    cv2.imshow("face_section",face_section)
    cv2.imshow("My video",frame)
    key_pressed = cv2.waitKey(1) & 0xFF  ## it is going to take the key from keyboard for quit
    # by tyhe help of bit wise conversion
    if key_pressed == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

print("real shape of face_data before array conversion",len(face_data),type(face_data[0]))
face_data=np.array(face_data)
# convert the faces list into numpy error
face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))


np.save(f"./data/{name}.npy",face_data)
print("data collected successfully")


      
   