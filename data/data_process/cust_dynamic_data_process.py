import os
from glob import glob
import numpy as np
from PIL import Image
import shutil

def renamelabel(label_path):
    labels=glob(label_path+'/*.txt')
    for name in labels:
        os.rename(name,name.replace('.xml',''))


def getLabel(label_path,split_ratio=0.2):
    labels=glob(label_path+'/*.txt')
    tot_num=len(labels)
    print('tot_num=',tot_num)
    label_rand_idx=np.arange(tot_num)
    np.random.shuffle(label_rand_idx)
    val_idx=label_rand_idx[:int(tot_num*split_ratio)]
    train_idx=label_rand_idx[int(tot_num*split_ratio):]
    val_label=[labels[i] for i in val_idx]
    train_label=[labels[i] for i in train_idx]
    return train_label,val_label


def getCustData_yolo(cust_dataset,target_path_label,labels):
    if os.path.exists(target_path_label):
        shutil.rmtree(target_path_label)
    os.makedirs(target_path_label)
    
    target_path_img=target_path_label.replace('labels','images')
    if os.path.exists(target_path_img):
        shutil.rmtree(target_path_img)
    os.makedirs(target_path_img)


    for label_name in labels:
        name=os.path.basename(label_name)
        target_label=os.path.join(target_path_label,name)
        if not os.path.exists(target_label):
            shutil.copy(label_name,target_label)
        
        img_name=label_name.replace('.txt','.jpg')
        name=os.path.basename(img_name)
        target_img=os.path.join(target_path_img,name)
        if not os.path.exists(target_img) and os.path.exists(img_name):
            shutil.copy(img_name,target_img)

    
if __name__=="__main__":
    cust_dataset='/home/xinglong/project/yolov3/data/cust_data/7'
    target_path_root='/home/xinglong/project/yolov3/data/cust_data_7'
    split_ratio=0.4

    # renamelabel(cust_dataset)#rename orgdatset label.txt  xxx.xml.txt -> xxx.txt

    train_label,val_label=getLabel(cust_dataset,split_ratio)

    target_path_label_train=target_path_root+'/labels/train'
    getCustData_yolo(cust_dataset,target_path_label_train,train_label)

    target_path_label_val=target_path_root+'/labels/val'
    getCustData_yolo(cust_dataset,target_path_label_val,val_label)




