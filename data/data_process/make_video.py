#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import cv2

name = glob.glob('*.jpg')
name.sort()
print(name)
img = cv2.imread(name[0])
imginfo = img.shape
size = (imginfo[1], imginfo[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
videoWrite = cv2.VideoWriter('1.mp4', fourcc, 7.6, size)

for index in name:
   print(index)
   img = cv2.imread(index)
   videoWrite.write(img)


