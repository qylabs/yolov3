#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import glob
import cv2

name = glob.glob('*.jpg')
name.sort()
print(name)
i = 0
for index in name:
   img = Image.open(index)
   img = img.convert("L")
   img=img.resize((160,120))
   img.save(str(i)+'.ppm')
   i+=1
