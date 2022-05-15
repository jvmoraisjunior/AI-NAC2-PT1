#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from doctest import testfile
import cv2
import numpy as np

cap = cv2.VideoCapture("q1.mp4")

template = cv2.imread("AsDeOuro.png", 0)
w, h = template.shape[::-1]

object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    mask = object_detector.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 100:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()