import os
from glob import glob
import numpy as np
from PIL import Image
import shutil

def xyxy2xywhn(x,w, h):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.copy()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2 /w # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2 /h # y center
    y[:, 2] = (x[:, 2] - x[:, 0])/w # width
    y[:, 3] = (x[:, 3] - x[:, 1])/h  # height

    return y

def ImgFolder(img_txt,img_path,target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    for name in open(img_txt,'r'):
        name=name.replace('\n','')
        img_name=name+'.jpg'
        # print('img_names',img_names)
        img=os.path.join(img_path,img_name)
        # print('imgs ',imgs)
        target_img_path=os.path.join(target_path,img_name)
        if not os.path.exists(target_img_path):
            shutil.copy(img,target_path)
        else:
            print(target_img_path,' exists')
            
    print('ImgFolder Success!')
    

def LabelFolder(img_txt,label_path,target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    for name in open(img_txt,'r'):
        name=name.replace('\n','')
        label_name=name+'.jpg'+'.txt'
        label=os.path.join(label_path,label_name)
        target_label_path=os.path.join(target_path,label_name.replace('.jpg',''))
        if os.path.exists(label):
            shutil.copy(label,target_label_path)#label should always be rewrite
            
            
    print('LabelFolder Success!')



def formatLabels(label_path):
    labels=glob(label_path+'/*.txt')
    print('len(labels) ',len(labels))
    for label_path in labels:
        img_path=label_path.replace('labels','images').replace('.txt','.jpg')
        image=Image.open(img_path)
        w,h=image.size
        # label_txt_lines=
        #delet first lines
        with open(label_path,'r') as f:
            label_txts=f.readlines()
        with open(label_path,'w') as f:
            f.writelines(label_txts[1:])
        
        fbbox=np.loadtxt(label_path)
        if fbbox.ndim==1:#for some case, y.ndim=1
            fbbox=fbbox[np.newaxis,:]
        # print(fbbox.shape)
        xywhn=xyxy2xywhn(fbbox[:,1:],w,h)
        fbbox[:,1:]=xywhn
        fbbox[:,0]=fbbox[:,0]-1 #widerperson use label 1-5
        # print('fbbox new ',fbbox)
        np.savetxt(label_path,fbbox,fmt=['%d', '%10.5f', '%10.5f', '%10.5f', '%10.5f'])



if __name__=="__main__":
    data_root='/home/xinglong/project/data/WiderPerson'
    
    ##Build train images
    img_txt=data_root+'/train.txt'
    img_path=data_root+'/Images'
    target_path='./images/train'
    ImgFolder(img_txt,img_path,target_path)

    # ###Build val images
    img_txt=data_root+'/val.txt'
    img_path=data_root+'/Images'
    target_path='./images/val'
    ImgFolder(img_txt,img_path,target_path)

    # ###Build train labels
    img_txt=data_root+'/train.txt'
    label_path=data_root+'/Annotations'
    target_path='./labels/train'
    LabelFolder(img_txt,label_path,target_path)
    
    ###Build val labels
    img_txt=data_root+'/val.txt'
    label_path=data_root+'/Annotations'
    target_path='./labels/val'
    LabelFolder(img_txt,label_path,target_path)

    ####Update label format to yolo format
    label_path='./labels/train'
    formatLabels(label_path)


    label_path='./labels/val'
    formatLabels(label_path)





