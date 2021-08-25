#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import glob
import cv2
import os

def convert_to_ppm(img_path,size,target_path):
   names = glob.glob(img_path+'/*.jpg')
   for index in names:
      img_name=os.path.basename(index)
      img = Image.open(index)
      img = img.convert("L")
      
      img=img.resize(size)
      
      target=os.path.join(target_path,img_name.replace('.jpg','.ppm'))
      # print(target)
      img.save(target)

if __name__=="__main__":
   # img_path='data/cust_data/img_7'
   # size=(160,128)
   # target_path='data/cust_data/img_7_ppm'
   # convert_to_ppm(img_path,size,target_path)

   img_path='cust_data/img_3'
   size=(160,128)
   target_path='cust_data/img_3_ppm'
   convert_to_ppm(img_path,size,target_path)
