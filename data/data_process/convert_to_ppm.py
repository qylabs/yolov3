#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import glob
# import cv2
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
   
def convert_to_jpg(img_path,size,target_path):
   names = glob.glob(img_path+'/*.ppm')
   print('names: ',names)
   for index in names:
      img_name=os.path.basename(index)
      img = Image.open(index)
      if size:
         img=img.resize(size)
      
      target=os.path.join(target_path,img_name.replace('.ppm','.jpg'))
      # print(target)
      img.save(target)
   print("Done!")

if __name__=="__main__":
   # img_path='data/cust_data/img_7'
   # size=(160,128)
   # target_path='data/cust_data/img_7_ppm'
   # convert_to_ppm(img_path,size,target_path)

   # img_path='cust_data/img_3'
   # size=(160,128)
   # target_path='cust_data/img_3_ppm'
   # convert_to_ppm(img_path,size,target_path)

   for idx in [6,7,8]:
      img_path='cust_data/{}_100_results_origin'.format(str(idx))
      print('img_path:',img_path)
      size=None
      target_path=img_path+'_jpg'
      if not os.path.exists(target_path):
         os.makedirs(target_path)
      convert_to_jpg(img_path,size,target_path)
