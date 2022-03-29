import cv2
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



while True:
    ret,frame =  cam.read()
    if ret == False:
         continue 

    faces=model.detectMultiScale(frame) # it will take number of faces detected equal to that many records
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) # 2 pixel  thickness


    cv2.imshow("My video",frame)
    key_pressed = cv2.waitKey(1) & 0xFF  ## it is going to take the key from keyboard for quit
    # by tyhe help of bit wise conversion
    if key_pressed == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
   




 
      
   