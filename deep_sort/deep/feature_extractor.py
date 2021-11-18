import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from model import Net
from modellib import build_model

class Extractor(object):
    def __init__(self, 
                net, 
                image_size=(256, 128),
                pixel_mean=[0.485, 0.456, 0.406],
                pixel_std=[0.229, 0.224, 0.225],
                pixel_norm=True,
                **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net=net.to(self.device)
        self.net.eval()
        self.size = image_size
        
        # Build transform functions
        transforms_list = [transforms.Resize(self.size),transforms.ToTensor()]
        if pixel_norm:
            transforms_list+=[transforms.Normalize(mean=pixel_mean,std=pixel_std)]
        self.norm = transforms.Compose(transforms_list)

    def _preprocess(self, im_crops):
        im_batch = torch.cat([self.norm(im).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("test_sample/0003_c1s6_015971_00.jpg")[:,:,(2,1,0)]

    model_name='resnet18'
    num_classes=751
    in_channel=3
    checkpoint='checkpoint/resnet18.pt'
    # net definition
    if model_name:
        print('use model: ',model_name)
        net=build_model(model_name,num_classes=num_classes, pretrained=True,in_channel=in_channel,reid=True)
    else:
        net = Net(num_classes=num_classes,reid=True)
    
    checkpoint = torch.load(checkpoint)
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict, strict=False)
    net.eval()

    extr = Extractor(net)
    feature = extr(img)
    print(feature.shape)

