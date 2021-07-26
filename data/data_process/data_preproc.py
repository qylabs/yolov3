import os
from glob import glob
import numpy as np
from PIL import Image

def img2gray(img_folder):
    #change RGB img to Gray img
    img_list=glob(img_folder)
    for idx,img_path in enumerate(img_list):
        print('idx: ',idx)
        img=Image.open(img_path)
        # print('img.mode ',img.mode)
        img= img.convert('L')
        img.save(img_path)
    print('conver rgb to gray Success!')


if __name__=="__main__":
    #remember to copy original Dataset. This code will modify dataset images directly
    #cp -r WiderPerson WiderPerson_gray
    data_root='/home/xinglong/project/data/WiderPerson_gray'

    img_folder=data_root+'/Images/*.jpg'
    img2gray(img_folder)

