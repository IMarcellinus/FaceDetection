'''
langkah-langkah :
python -m pip install opencv-contrib-python
pip install Pilow
deteksi wajah => https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
deteksi mata => https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
langkah untuk face recegnition: rekam data waja, training data wajah, recegnition
'''

import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 640) # ubah lebar cam code 3 dan 640 itu ukuran
cam.set(4, 480) # ubah tinggi cam mode 4 dan 480 itu ukuran
faceDetector = cv2.CascadeClassifier('deteksiMuka.xml')
eyesDetector = cv2.CascadeClassifier('deteksiMata.xml')
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5) #frame, scaleFactor, minNumbers
    eyes = eyesDetector.detectMultiScale(abuAbu, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2) # x + w yaitu mengukur lebar dengan sumbu w dan y + h mengukur lebar dengan h dengan sumbu y. warna dengan settingan rgb yaitu 0,0,255 dengan warna merah. angka 2 artinya seberapa tebal.
    for (x, y, w, h) in eyes:
        # x + w yaitu mengukur lebar dengan sumbu w dan y + h mengukur lebar dengan h dengan sumbu y. warna dengan settingan rgb yaitu 0,0,255 dengan warna merah. angka 2 artinya seberapa tebal.
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('Webcamku', frame)
    cv2.imshow('Webcamku 2', abuAbu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
