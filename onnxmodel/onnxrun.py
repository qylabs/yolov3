import os
import numpy as np
import onnxruntime as ort
import cv2
import copy
import torch
import torchvision

strides=[16,32] #

anchors=[[10,14, 23,27, 37,58]  # P4/16
        ,[81,82, 135,169, 344,319]]  # P5/32

np.set_printoptions(threshold=np.inf)

def NMS(dets,threshold):
    '''
    dets=np.array([(x1,y1,x2,y2,scores)])

    dets = np.array([[210, 30, 280, 5, 0.6],
	                 [120, 210, 240, 110, 1],
                   [70, 150, 260, 120, 0.8],
					[200, 180, 360, 140, 0.7]])
	threshold=0.1
	keep_dets=NMS(dets,threshold)
    
    print('keep_dets ',keep_dets)
	print(dets[keep_dets])

    '''
	#(x1、y1）（x2、y2）为box的左上和右下角标
    x1=dets[:,0]
    y1=dets[:,1]
    x2=dets[:,2]
    y2=dets[:,3]
    scores=dets[:,4]

    #areas of every bbox
    area=(x2-x1)*(y2-y1)
    # print('area ',area)
    #sort the idx rather val
    order=scores.argsort()[::-1]#::-1 逆序
    print('order ',order)

    tmp=[]
    while order.size>0:
        i=order[0]
        tmp.append(i)#save the order idx
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1=np.maximum(x1[i],x1[order[1:]])
        yy1=np.minimum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]])
        yy2=np.maximum(y2[i],y2[order[1:]])
        
        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w=np.maximum(0.0,xx2-xx1)
        h=np.maximum(0.0,yy2-yy1)
        inter=w*h
        # print('inter ',inter,inter.shape)
        #iou
        iou=inter/(area[i]+area[order[1:]]-inter)
        print('iou ',iou)
        
        #找到重叠度不高于阈值的矩形框索引
        idx=np.where(iou<=threshold)[0] #this idx is iou idx, 1 smaller than org dets
        #print('idx ',idx)

        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        # print('before ',order)
        order = order[idx+1]
        # print('after order ',order)
    return tmp


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def plot_one_box(x, im, img_save_path,color=(128, 128, 128), label=None, line_thickness=1):
    '''
    x is xyxy
    '''
    # Plots one bounding box on image 'im' using OpenCV
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    print('bbox c1,c2',c1,c2)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    cv2.imwrite(img_save_path,im)

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300,iter=True):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    nc = prediction.shape[2] - 5  # number of classes
    # print('prediction.shape:',prediction.shape)
    # print('nc=',nc)
    # nc=7
    xc = prediction[..., 4] > conf_thres  # candidates
    # prediction=prediction[xc] #filter out low confidence
    # print('xc=',xc)
    # print('xc true',np.argwhere(xc))
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 3000  # maximum number of boxes
    # time_limit = 10.0  # seconds to quit after
    # redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # print('multi_label is:', multi_label)
    merge = False  # use merge-NMS

    ##support batch-images
    output=[[]]*prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # print('0 x.shape',x.shape)
        x = x[xc[xi]]  # confidence
        # print('1 x=',x)
        # Cat apriori labels if autolabelling
        # if labels and len(labels[xi]):
        #     l = labels[xi]
        #     v = torch.zeros((len(l), nc + 5), device=x.device)
        #     v[:, :4] = l[:, 1:5]  # box
        #     v[:, 4] = 1.0  # conf
        #     v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
        #     x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue
        # print('1 x.shape ',x.shape)

        if iter:
            x[:,:4] = xywh2xyxy(x[:, :4])
        else:
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # print('1-2 x=',x)
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            # print('1 box=', box)

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                """ nnn = (x[:, 5:] > conf_thres)
                print('nnn is: ', nnn)
                nnnnn = nnn.nonzero()
                print('nnnnn is:', nnnnn)
                nnnnn2 = nnn.nonzero(as_tuple=False)
                print('nnnnn2 is:', nnnnn2)
                j, i = nnnnn
                print('i = ', i, 'j = ', j) """
                j, i = (x[:, 5:] > conf_thres).nonzero() #.T  #as_tuple=False
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf = x[:, 5:].max(1, keepdims=True)
                # print('conf =', conf)
                j=x[:,5:].argmax(1).reshape(conf.shape) #cls num
                x = np.concatenate((box, conf, j.astype(conf.dtype)), 1)
                # print('1-3 x=',x)
                conf_mask=x[:,4]>conf_thres
                x=x[conf_mask]
                # print('1-4 x=',x)
        # print('2x',x)
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        
        """ # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded """
        class_n = set(x[:,5]) 
        print('class_n is:', class_n)
        len_c = len(class_n)

        if len_c == 1:

            keep_dets=NMS(x,iou_thres)
            # print('keep_dets',keep_dets)
            output[xi]=x[keep_dets]
            # print('3 output is:', output)

        else:
            x_c1 = np.zeros((int(x.shape[0]), int(x.shape[1])))
            # x_c1 = [[]]
            x_c2 = np.zeros((int(x.shape[0]), int(x.shape[1])))
            # print(' before x_c1 = ', x_c1)
            # print(' before x_c2 = ', x_c2)
            class_1 = x[0,5]
            # print('class_l = ',class_1)
            index = -1
            for j in x[:,5]:
                
                index += 1
                # print('index = ', index)
                if j == class_1:
                    """ print('j = ', j)
                    print('x_c1 = ', x_c1)
                    print('x[index,:] = ', x[index,:])
                    print('x_c1[index,:] = ', x_c1[index,:]) """
                    aaaa = x[index,:]
                    # print('aaaa = ', aaaa)

                    x_c1[index,:] += aaaa
                    # print('x_c1[index,:] = ', x_c1[index,:])
                    # x_c1 =[x_c1, x[index,:]]
                    # x_c1.append(np.mat(x[index,:]))
                    # print('x_c1 = ', x_c1)
                else:
                    """ print("!=")
                    print('x_c2 = ', x_c2)
                    print('x[index,:] = ', x[index,:])
                    print('x_c2[index,:] = ', x_c2[index,:]) """
                    aaaab = x[index,:]
                    x_c2[index,:] = aaaab
                    # x_c2 = [x_c2, x[index,:]]
                    # print('x_c2 = ', x_c2)
            # print('x_c1 = ', x_c1)
            # print('x_c2 = ', x_c2)
            xc1 = x_c1[..., 4] > conf_thres
            x_c1 = x_c1[xc1]
            xc2 = x_c2[..., 4] > conf_thres
            x_c2 = x_c2[xc2]
            # print('x_c1 = ', x_c1)
            # print('x_c2 = ', x_c2)

            keep_dets1=NMS(x_c1,iou_thres)
            # print('keep_dets1',keep_dets1)
            keep_dets2=NMS(x_c2,iou_thres)
            # print('keep_dets2',keep_dets2)
            output1 = x_c1[keep_dets1]
            output2 = x_c2[keep_dets2]
            # print('output1 = ', output1)
            # print('output2 = ', output2)
             
            output_ = np.vstack([output1, output2])
            # print('output_ is:', output_)
            output[xi] = output_
            # print('3 output is:', output)


        
        # print('4 output.shape',output.shape)

    return output


def transfrom_img(img_path,gray_input=False):
    img=cv2.imread(img_path)
    final=np.transpose(img,(2,0,1)).astype(np.float32) #BGR2RGB
    final = np.ascontiguousarray(final)
    
    if gray_input:
        #gray=0.2989*r+0.5870*g+0.1140*b
        final2=0.2989*final[0,:,:]+0.5870*final[1,:,:]+0.1140*final[2,:,:]
        final=final2[np.newaxis,:]
    
    final /= 255.0  # 0 - 255 to 0.0 - 1.0
    if final.ndim == 3:
        final=np.expand_dims(final,axis=0)
    
    return final


def onnxrun(onnx_path,input_array):
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_array}
    ort_outs = ort_session.run(None, ort_inputs)
    print('len(ort_outs) ',len(ort_outs))
    return ort_outs


def yolo_proc(img_path,onnx_path,conf_thres=0.2,iou_thres=0.45):
    '''
    single one img
    '''
    img0=cv2.imread(img_path)
    #preproc img
    input_array=transfrom_img(img_path,gray_input=True)
    print(input_array.shape)
    out=onnxrun(onnx_path,input_array)[0] #onnx-graph has only 1 output
    # print('==>>out.shape ',out.shape)
    #postproc 
    #nms out
    pred = non_max_suppression(out, conf_thres,iou_thres)
    print('==>>pred_nms ',pred)

    #plot
    img_save_path=img_path+'.jpg'
    print(img_save_path)
    for i, det in enumerate(pred):  # detections per image
        print(i,', det',det)
        for *xyxy, conf, cls in det:
            print('xyxy ',xyxy)
            c = int(cls)  # integer class
            label_mark=f'{c}-{conf:.2f}'
            print('label_mark ',label_mark)
            plot_one_box(xyxy, img0, img_save_path,color=(2, 8, 255),label=label_mark)
    print('end')


#######################################################
##the following code is to manually add postproc(anchors,grid,reshape etc) to obtain bbox

def make_grid(nx=20,ny=20):
    '''
    nx,ny=feature_shape
    '''
    yv, xv = np.meshgrid(np.arange(ny), np.arange(nx)) #np.meshgrid result is transpose of torch.meshgrid
    grid=np.stack((xv.T, yv.T), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32) #center point grid
    return grid


def make_anchor_grid(anchors):
    '''
    anchor is list,len(anchor)=head_num

    yolov3-tiny anchors:
        anchors=[[10,14, 23,27, 37,58]  # P4/16
                ,[81,82, 135,169, 344,319]]  # P5/32
    
    feature shape=(bs,na,nx,ny,no)
        no: number of outputs per anchor, class+confidence+bbx
        na: number of anchors
        (nx,ny): feature size
        bs: batch_size

    '''
    nl = len(anchors)  # number of detection layers
    na = len(anchors[0]) // 2  # number of anchors
    a = np.array(anchors).reshape(nl, -1, 2).astype(np.float32) #anchor reshaped as (nl=2,na=3,2), 2 head,3anchors,2 width/height of anchors
    anchor_grid=a.reshape(nl, 1, -1, 1, 1, 2) # shape(nl,1,na,1,1,2), align with nl, then 1(for pred_wh broadcast), then na, then (1,1) then 2(width,height)
    return anchor_grid


def center2grid(output,feat_nx,feat_ny,nc,anchors,strides):
    '''
    len(output)==nl
    '''
    bs=1 #batch_size
    nl=len(anchors) #num head
    na=len(anchors[0])//2 #num anchor
    no=nc+5 #num output
    anchor_grid=make_anchor_grid(anchors)
    # print('anchor_gird.shape ',anchor_grid.shape)

    grid=[[]]*nl
    z=[]
    for i in range(nl):
        y=output[i].reshape(bs,na,feat_ny[i],feat_nx[i],no) #reshape onnx_sim back to (bs,na,ny,nx,no)
        grid[i]=make_grid(feat_nx[i],feat_ny[i])
        # print('grid[i].shape ',grid[i].shape,grid)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * strides[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

        z.append(y.reshape(bs, na*feat_ny[i]*feat_nx[i], no))
        
    return np.concatenate(z,axis=1)


def yolo_proc_simp(img_path,onnx_path,strides,anchors,nc,conf_thres=0.2,iou_thres=0.45):
    '''
    single one img
    '''
    img0=cv2.imread(img_path)
    input_shape=img0.shape
    # print('input_shape',input_shape)
    #preproc img
    input_array=transfrom_img(img_path,gray_input=True)
    print(input_array.shape)
    out=onnxrun(onnx_path,input_array) #onnx-graph has only 1 output
    print('onnx out.shape ',len(out))

    #postproc 
    #1. center2grid of onnx_sim_2H model
    feat_nx=[int(input_shape[1]/s) for s in strides]
    feat_ny=[int(input_shape[0]/s) for s in strides]
    # print('feat_nx,feat_ny ',feat_nx,feat_ny)
    out=center2grid(out,feat_nx,feat_ny,nc,anchors,strides) #obtain the pred
    # print('==>>final out.shape',out.shape)
    
    # #2. nms out
    # pred = non_max_suppression(out, conf_thres,iou_thres)
    # print('pred_nms ',pred)

    # # #plot
    # img_save_path=img_path+'.jpg'
    # print(img_save_path)
    # for i, det in enumerate(pred):  # detections per image
    #     print(i,', det',det)
    #     for *xyxy, conf, cls in det:
    #         # print('xyxy ',xyxy)
    #         c = int(cls)  # integer class
    #         label_mark=f'{c}-{conf:.2f}'
    #         # print('label_mark ',label_mark)
    #         plot_one_box(xyxy, img0, img_save_path,color=(2, 8, 255),label=label_mark)
    # print('end')


def yolo_proc_simp_1H(img_path,onnx_path,strides,anchors,nc,conf_thres=0.2,iou_thres=0.45):
    '''
    single one img
    '''
    img0=cv2.imread(img_path)
    input_shape=img0.shape
    # print('input_shape',input_shape)
    #preproc img
    input_array=transfrom_img(img_path,gray_input=True)
    print(input_array.shape)
    print('yolo_1H ==>> input: ',input_array)
    out1=onnxrun(onnx_path,input_array) #onnx-graph has only 1 output
    np.set_printoptions(threshold=np.inf)
    print('onnx out1.shape ',len(out1),out1[0])
    print("out[0].min() ",out1[0].min())

    out=[out1[0][:,:240,:],out1[0][:,240:,:]]
    #postproc 
    #1. center2grid of onnx_sim_2H model
    feat_nx=[int(input_shape[1]/s) for s in strides]
    feat_ny=[int(input_shape[0]/s) for s in strides]
    # print('feat_nx,feat_ny ',feat_nx,feat_ny)
    out=center2grid(out,feat_nx,feat_ny,nc,anchors,strides) #obtain the pred
    print('==>>final out.shape',out.shape)
    
    # #2. nms out
    pred = non_max_suppression(out, conf_thres,iou_thres)
    print('pred_nms ',pred)

    # #plot
    img_save_path=img_path+'.jpg'
    print(img_save_path)
    for i, det in enumerate(pred):  # detections per image
        print(i,', det',det)
        for *xyxy, conf, cls in det:
            # print('xyxy ',xyxy)
            c = int(cls)  # integer class
            label_mark=f'{c}-{conf:.2f}'
            # print('label_mark ',label_mark)
            plot_one_box(xyxy, img0, img_save_path,color=(2, 8, 255),label=label_mark)
    print('end')


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def center2grid2(output,feat_nx,feat_ny,nc,anchors,strides):
    '''
    output[i]=(bs,na*no,feat_ny[i],feat_nx[i])

    reshape+trasnpose+sigmoid

    len(output)==nl
    '''
    bs=1 #batch_size
    nl=len(anchors) #num head
    na=len(anchors[0])//2 #num anchor
    no=nc+5 #num output
    anchor_grid=make_anchor_grid(anchors)
    print('anchor_gird.shape ',anchor_grid.shape)
    print('anchor_gird\n ',anchor_grid)

    grid=[[]]*nl
    z=[]
    for i in range(nl):
        tmp=np.ascontiguousarray(output[i].reshape(bs,na,no,feat_ny[i],feat_nx[i]).transpose((0,1,3,4,2)))
        # print(i,', tmp.shape ',tmp.shape)
        # print('after reshape and transpose ')
        tmp=sigmoid(tmp)
        # print(i, ', tmp after sigmoid \n',tmp)

        y=tmp
        grid[i]=make_grid(feat_nx[i],feat_ny[i])
        # print('grid[i].shape ',grid[i].shape,grid)
        # print("before y[..., 0:2]=",y[..., 0:2])
        # print("before y[..., 2:4]=",y[..., 2:4])

        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * strides[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        # print("after y[..., 0:2]=",y[..., 0:2])
        # print("after y[..., 2:4]=",y[..., 2:4])
        
        z.append(y.reshape(bs, na*feat_ny[i]*feat_nx[i], no))
        # print('z is:', z)
        
    return np.concatenate(z,axis=1)

def center2grid2_iter(output,feat_nx,feat_ny,nc,anchors,strides,conf_thred=0.1):
    '''
    this iter method use 3 for loops to achieve similar center2grid2 method
    align with nn_gap8_project yolo postproc

    output[i]=(bs,na*no,feat_ny[i],feat_nx[i])

    reshape+trasnpose+sigmoid

    len(output)==nl
    '''
    bs=1 #batch_size
    nl=len(anchors) #num head   2
    # print('nl is:', nl) 
    na=len(anchors[0])//2 #num anchor   3
    # print('na is:', na)
    no=nc+5 #num output
    anchor_grid=make_anchor_grid(anchors)
    # print('anchor_gird.shape ',anchor_grid.shape)
    # print('anchor_gird\n ',anchor_grid)
    feat_nx0=copy.deepcopy(feat_nx)     #[8,4]
    feat_ny0=copy.deepcopy(feat_ny)     #[10,5]
    
    z=[]
    
    out=[]
    for i in range(nl):
        print('================idx=',i)
        output_flat=output[i].flatten() #align with c code
        feat_nx=int(feat_nx0[i])      #8
        feat_ny=int(feat_ny0[i])      #10
        feat_map=feat_nx*feat_ny
        print('feat_map=',feat_map)
        # print('after sigmoid out_sigmoid=',sigmoid(output_flat))
        for a in range(na):
            for feat in range(feat_map):
                # print(">>>a=",a,', feat=',feat)
                max_cls=0
                cls_out=0
                for o in range(5,no):
                    tmp_cls=output_flat[a*no*feat_map+o*feat_map+feat]
                    if tmp_cls>max_cls:
                        max_cls=tmp_cls
                        cls_out=o-5
                        # print('cls ',cls_out)
                # print('before sigmoid score=',output_flat[a*no*feat_map+4*feat_map+feat],output_flat[a*no*feat_map+(cls_out+5)*feat_map+feat])
                score=sigmoid(output_flat[a*no*feat_map+4*feat_map+feat])
                # print('score=',score)
                if score>conf_thred:
                    # print('candicate score=',score)
                    # print('x idx=',a*no*feat_map+0*feat_map+feat)
                    # print("before sigmoid x=",output_flat[a*no*feat_map+0*feat_map+feat],', y=',output_flat[a*no*feat_map+1*feat_map+feat])
                    x=sigmoid(output_flat[a*no*feat_map+0*feat_map+feat])
                    y=sigmoid(output_flat[a*no*feat_map+1*feat_map+feat])
                    # print("after sigmoid x=",x,", y=",y)
                    grid_x_shift=feat%feat_nx
                    grid_y_shift=feat//feat_nx
                    # print('grid_x_shift=',grid_x_shift,', grid_y_shift=',grid_y_shift)
                    x=(x*2-0.5+grid_x_shift)*strides[i]
                    y=(y*2-0.5+grid_y_shift)*strides[i]
                    # print('final x=',x,', final y=',y)


                    # print("before sigmoid w=",output_flat[a*no*feat_map+2*feat_map+feat],', h=',output_flat[a*no*feat_map+3*feat_map+feat])
                    w=sigmoid(output_flat[a*no*feat_map+2*feat_map+feat])
                    h=sigmoid(output_flat[a*no*feat_map+3*feat_map+feat])
                    # print("after sigmoid w=",w,", h=",h)
                    w=w*w*4*anchors[i][2*a+0]
                    h=h*h*4*anchors[i][2*a+1]
                    # print('final w=',w,', final h=',h)

                    score*=sigmoid(output_flat[a*no*feat_map+(cls_out+5)*feat_map+feat])
                    z.append([x,y,w,h,score,cls_out])
                    # print('z is:', z)
        print('len(z)=',len(z))
        
    return np.concatenate(z)



def yolo_proc_simp_2H_noreshape_nosigmoid(img_path,onnx_path,strides,anchors,nc,conf_thres=0.2,iou_thres=0.45,iter=False):
    '''
    2Head output noreshape_nosigmoid
    sigmoid reshape happened here
    '''
    img0=cv2.imread(img_path)
    input_shape=img0.shape
    # print('input_shape',input_shape)
    #preproc img
    input_array=transfrom_img(img_path,gray_input=True)  #(128,160)
    print(input_array.shape)
    out=onnxrun(onnx_path,input_array) #onnx-graph has only 1 output
    # print('out is: ', out)
    print('==>> 2H output onnx out.shape ',len(out))
    print("out[0].shape, out[1].shape ",out[0].shape, out[1].shape)
    # print("out[0]==\n",out[0])
    # print("out[1]==\n",out[1])
    

    #postproc 
    #0. reshape+sigmoid
    #1. center2grid of onnx_sim_2H model
    feat_nx=[int(input_shape[1]/s) for s in strides]  #strides[16,32] [8, 4]
    feat_ny=[int(input_shape[0]/s) for s in strides]  #               [10,5]
    # print('feat_nx,feat_ny ',feat_nx,feat_ny)
    if iter is False:
        out1=center2grid2(out,feat_nx,feat_ny,nc,anchors,strides) #obtain the pred
        print('==>>final out1.shape',out1.shape)
        # print("out1 ",out1)
        out=out1
    else:
        print("===================Second method====================")
        out2=center2grid2_iter(out,feat_nx,feat_ny,nc,anchors,strides).reshape(1,-1,6) #obtain the pred
        print('==>>final out2.shape',out2.shape)
        # print("out2 ",out2)
        out=out2

    #2. nms out
    pred = non_max_suppression(out, conf_thres,iou_thres,iter=False, multi_label=False)
    # print('pred_nms ',pred)

    # #plot
    img_save_path=img_path+'.jpg'
    print(img_save_path)
    for i, det in enumerate(pred):  # detections per image
        print(i,', det',det)
        for *xyxy, conf, cls in det:
            # print('index is:', index)
            print('xyxy ',xyxy)
            c = int(cls)  # integer class
            label_mark=f'{c}-{conf:.2f}'
            print('label_mark ',label_mark)
            plot_one_box(xyxy, img0, img_save_path,color=(2, 8, 255),label=label_mark)
    print('end')




if __name__=="__main__":
    # onnx_path='./weights/yolov3-tiny_sim.onnx'
    # onnx_path='./weights/yolov3-tiny_sim_relu.onnx'
    # onnx_path='./weights/onnx_model_zoo/tiny-yolov3-11.onnx'
    # onnx_path='./runs/train/exp_yolov3_tiny3_gray_WP/weights/best_org.onnx'
    

    # img_path='/home/shihuiyu/dataset/origin_figures/img_OUT_100_6.ppm'
    img_file='/home/shihuiyu/dataset/qiyuan_gesture_cleaned_resized/test/person15-l/'
    img_path_list = os.listdir(img_file)

    # compute ONNX Runtime output prediction
    batch_size=1
    channels=1#3
    height=128 #320
    width=160 #320

    # # input_array=np.random.randn(batch_size, channels, height, width).astype(np.float32)
    # input_array=transfrom_img(img_path,gray_input=True)
    # print(input_array.shape)
    
    # out=onnxrun(onnx_path,input_array) #onnx-graph has only 1 output
    # print('out.shape ',len(out))
    # print("out ",out[0].shape,out[1].shape)


    # yolo_proc(img_path,onnx_path,conf_thres=0.2,iou_thres=0.45)
    
    # print("==============================")
    # onnx_path='./runs/train/exp_yolov3_tiny3_gray_WP/weights/best_simp_2H.onnx'
    # yolo_proc_simp(img_path,onnx_path,strides,anchors,nc=5,conf_thres=0.2,iou_thres=0.45)

    # onnx_path='./runs/train/exp_yolov3_tiny3_gray_WP/weights/best.onnx'
    # yolo_proc_simp_1H(img_path,onnx_path,strides,anchors,nc=5,conf_thres=0.2,iou_thres=0.45)

    print("==============================")
    # onnx_path='./runs/train/exp24/weights/best.onnx'
    onnx_path='/home/shihuiyu/yolov3-master/runs/train/exp28/weights/best.onnx'

    for imag_path in img_path_list:
        img_path = img_file + imag_path
        print('img_path is:', img_path)
        yolo_proc_simp_2H_noreshape_nosigmoid(img_path,onnx_path,strides,anchors,nc=7,conf_thres=0.2,iou_thres=0.2,iter=False)  #iou 0.45

