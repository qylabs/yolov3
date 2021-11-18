#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#用于标签修改
from tqdm import tqdm
import os

dir = '../..//data/tmp/labels/train'
#  获取文件夹内的文件名
FileNameList = os.listdir(dir)
cnt = 0
total = len(FileNameList)

remain_str = ['0','1','2','3']
for file in tqdm(FileNameList):
   if(file.find('.txt') >= 0):
        name = dir + os.sep +  file
        file_data = ''
        with open(name, "r") as f:
            for line in f:
                if line[0] in remain_str:
                    file_data += line
                else:
                    print(line)
            #print(file_data)

        with open(name,"w",encoding="utf-8") as f:
            f.write(file_data)
