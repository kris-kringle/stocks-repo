import campy as cam
import cv2

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

my_cam = cam.camera()

while True:
    img, timestamp = my_cam.getFrame()
    cv2.imshow('image', img)
    cv2.waitKey(1)

    # Q to Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

my_cam.stopCam()