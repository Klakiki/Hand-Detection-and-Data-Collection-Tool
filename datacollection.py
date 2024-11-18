import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands= 1)
offset = 20
imgSize = 300
counter = 0

folder = 'Data/Thank you'

# ตรวจสอบและสร้างโฟลเดอร์ ถ้ายังไม่มี
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    # อ่านรูปจาก webcam
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands :
        hand = hands[0]
        x,y,w,h = hand ['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset : y + h + offset, x-offset : x + w + offset]
        imgCropShape = imgCrop.shape

        aspectratio = h/w

        if imgCrop.size != 0:  # ตรวจสอบว่าภาพไม่ว่าง
            if aspectratio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        else:
            print("No hand detected or image is empty")

        if imgCrop.size != 0 and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            cv2.imshow('ImageCrop', imgCrop)
        else:
            print("Cannot display ImageCrop, image is empty or has invalid size")

        if imgWhite.size != 0 and imgWhite.shape[0] > 0 and imgWhite.shape[1] > 0:
            cv2.imshow('ImageWhite', imgWhite)
        else:
            print("Cannot display ImageWhite, image is empty or has invalid size")

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    # กด s ให้บันทึกรูปมือ
    if key == ord('s'):
        counter += 1
        file_path = os.path.join(folder, f'Image_{int(time.time())}.jpg')
        try:
            cv2.imwrite(file_path, imgWhite)
            print(f"Saved {file_path}")
            print(f"Counter: {counter}")
        except Exception as e:
            print(f"Error saving file: {e}")
            print(f"Error {file_path}")

    # กด q เพื่อออก
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()