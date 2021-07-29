import os
from glob import glob
from imgaug.augmenters.contrast import GammaContrast
from matplotlib.pyplot import axis

import numpy as np
import cv2


import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

def img_augment(img_path,draw_bbs=False):
    '''
    img_path should be in yolo_format  [cls,xn,yn,wn,hn]
    return:
        image_aug, is numpy array rgb mode, can be saved using cv2.imwrite(name,image_aug)
        bbs_aug, is BoundingBoxesOnImage
    '''
    image=cv2.imread(img_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bbs_path=img_path.replace('images','labels').replace('.jpg','.txt')
    img_shape=image.shape
    labels=np.loadtxt(bbs_path)
    w,h=img_shape[1],img_shape[0]

    bbs_xyxy=xywhn2xyxy(labels[:,1:],w,h)
    bbs_list=[BoundingBox(*bbs_xyxy[i]) for i in range(len(bbs_xyxy))]
    bbs = BoundingBoxesOnImage(bbs_list, shape=img_shape)

    #use iaa to implement image and obtain the bbs
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.3), # horizontally flip 30% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-40, 40), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 3),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    # iaa.BlendAlphaSimplexNoise(iaa.OneOf([
                    #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    # ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.BlendAlphaFrequencyNoise(
                            exponent=(-4, 0),
                            foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                            background=iaa.LinearContrast((0.5, 2.0))
                        )
                    ]),
                    iaa.OneOf([
                        iaa.LinearContrast((0.5, 1.8), per_channel=0.5), # improve or worsen the contrast
                        iaa.GammaContrast((0.5, 1.8))
                    ]),

                    # iaa.Grayscale(alpha=(0.0, 1.0)),
                    # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                    iaa.MotionBlur(k=[3,7,15],angle=(0,20)),
                ],
                random_order=True
            )
        ],
        random_order=True
    )

    #apply augmentation
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    
    if draw_bbs:
        image_aug=bbs_aug.draw_on_image(image_aug)
    
    # cv2.imwrite('test.jpg',image_aug)
    return image_aug,bbs_aug


def main(img_path,target_path,aug_scale=5):
    
    image_aug,bbs_aug=img_augment(img_path)



if __name__=="__main__":
    root='/home/xinglong/project/yolov3/'
    img_path=root+'data/WiderPerson_gray/images/train/'
    target_path=root+'data/cust_aug/'
    main(img_path,target_path)
