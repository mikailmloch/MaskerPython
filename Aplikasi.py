import cv2
import winsound
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

bw_threshold = 100

font = cv2.FONT_HERSHEY_SIMPLEX
org = (22, 22)
wearing_mask_font_color = (0, 255, 0)
not_wearing_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
wearing_mask_text = "Wearing Mask"
not_wearing_mask_text = "Not Wearing Mask"

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('Gui.ui', self)
        self.buttonCamera.clicked.connect(self.toggle_camera)

        self.cap = None
        self.camera_active = False
        self.count_1 = 0
        self.count_2 = 0


    def toggle_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(1)
            self.camera_active = True
            self.buttonCamera.setText("Stop Camera")
            self.update_camera()
        else:
            self.camera_active = False
            self.buttonCamera.setText("Start Camera")
            self.cap.release()
            self.labelCamera.clear()

    def update_camera(self):
        # mask_types = {
        #     "KF94": (0, 0, 255),  # Merah untuk KF94
        #     "Medical": (255, 0, 0),  # Biru untuk masker medis
        #     "Cloth": (0, 255, 0),  # Hijau untuk masker kain
        # }

        if self.camera_active:
            ret, img = self.cap.read()
            img = cv2.flip(img, 1)
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            cv2.imshow('grayscale', gray)
            black_and_white = np.where(gray >= bw_threshold, 255, 0).astype(np.uint8)
            cv2.imshow('black_and_white', black_and_white)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(img, 'Face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            if len(faces) == 0 and len(face_bw) == 0:
                cv2.putText(img, "Tidak ada Orang", org, font, font_scale, wearing_mask_font_color, thickness,cv2.LINE_AA)
                self.labelInfo.setText(wearing_mask_text)
            elif len(faces) == 0 and len(face_bw) == 1:
                cv2.putText(img, "Tidak ada Orang", org, font, font_scale, wearing_mask_font_color, thickness,cv2.LINE_AA)
                self.labelInfo.setText(wearing_mask_text)
            else:
                nose_rects = nose_cascade.detectMultiScale(gray, 1.5, 5)
                mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

                if len(mouth_rects) == 0 and len(nose_rects) == 0:
                    cv2.putText(img, wearing_mask_text, org, font, font_scale, wearing_mask_font_color, thickness,cv2.LINE_AA)
                    self.labelInfo.setText(wearing_mask_text)
                    self.labelHasil.setPixmap(self.convert_image(img))
                    self.labelHasil.setScaledContents(True)
                    self.labelHasil.resize(self.labelHasil.pixmap().size())

                    print("Wearing Masker " + str(self.count_1) + " saved")
                    file = 'D:\\tes masker\\' +"Wearing Masker " + str(self.count_1) + '.jpg'
                    cv2.imwrite(file, img)
                    self.count_1 += 1
                else:
                    for (mx, my, mw, mh) in mouth_rects:
                        if y < my < y + h:
                            cv2.putText(img, not_wearing_mask_text, org, font, font_scale,
                                        not_wearing_mask_font_color, thickness, cv2.LINE_AA)
                            self.labelInfo.setText(not_wearing_mask_text)
                            winsound.PlaySound('alarm.wav', winsound.SND_FILENAME)

                            # # Identify mask type based on color pattern
                            # roi_color = img[my:my + mh, mx:mx + mw]
                            # roi_hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
                            #
                            # mask_type = "Unknown"
                            # for mask_name, mask_color in mask_types.items():
                            #     lower_range = np.array([mask_color[0] - 10, 100, 100])
                            #     upper_range = np.array([mask_color[0] + 10, 255, 255])
                            #     mask = cv2.inRange(roi_hsv, lower_range, upper_range)
                            #     if cv2.countNonZero(mask) > 100:
                            #         mask_type = mask_name
                            #         break

                            self.labelHasil.setPixmap(self.convert_image(img))
                            self.labelHasil.setScaledContents(True)
                            self.labelHasil.resize(self.labelHasil.pixmap().size())
                            # self.labelInfo.setText(wearing_mask_text,f"Mask Type: {mask_type}")

                            print("Not Wearing Masker " + str(self.count_2) + " saved")
                            file = 'tes masker\\' +"Not Wearing Masker "+ str(self.count_2) + '.jpg'
                            cv2.imwrite(file, img)
                            self.count_2 += 1
                            break

            self.labelCamera.setPixmap(self.convert_image(img))
            self.labelCamera.setScaledContents(True)
            self.labelCamera.resize(self.labelCamera.pixmap().size())

        QApplication.processEvents()
        if self.camera_active:
            self.update_camera()

    def convert_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        image_qt = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(image_qt)

app = QApplication([])
window = MainWindow()
window.show()
app.exec_()