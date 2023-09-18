import face_recognition
import cv2
import numpy as np

# imgTiger = face_recognition.load_image_file('Images/Tiger.jpg')
# imgTiger = cv2.cvtColor(imgTiger,cv2.COLOR_BGR2RGB)
# imgTest = face_recognition.load_image_file('Images/Tiger1.jpg')
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
#
#
# fLocation = face_recognition.face_locations(imgTiger)[0]
# encodeT = face_recognition.face_encodings(imgTiger)[0]
# #cv2.rectangle(imgTiger,(fac))
# print(fLocation)
#
#
# cv2.imshow('Tiger',imgTiger)
# cv2.imshow('Tiger1',imgTest)
# cv2.waitKey(0)

imgT_bgr = face_recognition.load_image_file('Images/elon.jpg')
imgT_rgb = cv2.cvtColor(imgT_bgr,cv2.COLOR_BGR2RGB)
cv2.imshow('bgr', imgT_bgr)
cv2.imshow('rgb', imgT_rgb)
cv2.waitKey(0)

imgT =face_recognition.load_image_file('Images/elon.jpg')
imgT = cv2.cvtColor(imgT,cv2.COLOR_BGR2RGB)
#----------Finding face Location for drawing bounding boxes-------
face = face_recognition.face_locations(imgT_rgb)[0]
copy = imgT.copy()
#-------------------Drawing the Rectangle-------------------------
cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
cv2.imshow('copy', copy)
cv2.imshow('elon',imgT)
cv2.waitKey(0)


train_elon_encodings = face_recognition.face_encodings(imgT)[0]

# lets test an image
test = face_recognition.load_image_file('Images/elon.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(test)[0]
print(face_recognition.compare_faces([train_elon_encodings],test_encode))