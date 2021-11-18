#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#用于标签修改
from tqdm import tqdm
import os

dir = '/home/allen/data/widerperson_blur/labels/val'
#  获取文件夹内的文件名
FileNameList = os.listdir(dir)
cnt = 0
total = len(FileNameList)

old_str = ['1','2','3']
new_str = ['0','0','0']

for file in tqdm(FileNameList):
   if(file.find('.txt') >= 0):
        name = dir + os.sep +  file
        file_data = ''
        with open(name, "r") as f:
            for line in f:
                if line[0] in old_str:
                    idx = old_str.index(line[0])
                    line = new_str[idx] + line[1:]
                if line[0] in new_str:
                    file_data += line
                else:
                    print(line)

        with open(name,"w",encoding="utf-8") as f:
            f.write(file_data)
