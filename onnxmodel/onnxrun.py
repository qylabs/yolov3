import numpy as np
import onnxruntime as ort
import cv2

strides=[16,32] #

anchors=[[10,14, 23,27, 37,58]  # P4/16
        ,[81,82, 135,169, 344,319]]  # P5/32


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
    # print('order ',order)

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
        # print('iou ',iou)
        
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
    print('c1,c2',c1,c2)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    cv2.imwrite(img_save_path,im)

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    # prediction=prediction[xc] #filter out low confidence
    print('prediction.shape ',prediction.shape)
    
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 3000  # maximum number of boxes
    # time_limit = 10.0  # seconds to quit after
    # redundant = True  # require redundant detections
    # multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # merge = False  # use merge-NMS

    ##support batch-images
    output=[[]]*prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # print('0 x.shape',x.shape)
        x = x[xc[xi]]  # confidence

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
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf = x[:, 5:].max(1, keepdims=True)
            j=x[:,5:].argmax(1).reshape(conf.shape) #cls num
            x = np.concatenate((box, conf, j.astype(conf.dtype)), 1)
            conf_mask=x[:,4]>conf_thres
            x=x[conf_mask]

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

        # Batched NMS
        # c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # if i.shape[0] > max_det:  # limit detections
        #     i = i[:max_det]
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        # output[xi] = x[i]
        # if (time.time() - t) > time_limit:
        #     print(f'WARNING: NMS time limit {time_limit}s exceeded')
        #     break  # time limit exceeded
        keep_dets=NMS(x,iou_thres)
        # print('keep_dets',keep_dets)

        output[xi]=x[keep_dets]
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
    print('out.shape ',out.shape)
    #postproc 
    #nms out
    pred = non_max_suppression(out, conf_thres,iou_thres)
    print('pred_nms ',pred)

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




def make_grid(nx=20,ny=20):
    '''
    nx,ny=feature_shape
    '''
    yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
    grid=np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32) #center point grid
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


def center2grid(output,feat_nx,feat_ny,nc,anchors,stride):
    bs=1 #batch_size
    nl=len(anchors) #num head
    na=len(anchors[0])//2 #num anchor
    no=nc+5 #num output
    anchor_grid=make_anchor_grid(anchors)
    print('anchor_gird.shape ',anchor_grid.shape)
    grid=[[]]*nl
    z=[]
    for i in range(nl):
        y=output.reshape(bs,na,feat_ny[i],feat_nx[i],no) #reshape onnx_sim back to (bs,na,ny,nx,no)
        grid[i]=make_grid(feat_nx[i],feat_ny[i])
        print('grid[i].shape ',grid[i].shape)

        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

        z.append(y.view(bs, na*feat_ny[i]*feat_nx[i], no))
        
    return z




if __name__=="__main__":
    # onnx_path='./weights/yolov3-tiny_sim.onnx'
    # onnx_path='./weights/yolov3-tiny_sim_relu.onnx'
    # onnx_path='./weights/onnx_model_zoo/tiny-yolov3-11.onnx'
    # onnx_path='./runs/train/exp_yolov3_tiny3_gray_WP/weights/yolov3_tiny3_gray_WP.onnx'
    onnx_path='./runs/train/exp_yolov3_tiny3_gray_WP/weights/best.onnx'

    # compute ONNX Runtime output prediction
    batch_size=1
    channels=1#3
    height=128 #320
    width=160 #320

    # input_array=np.random.randn(batch_size, channels, height, width).astype(np.float32)
    input_array=transfrom_img('./img_OUT_0_resize.ppm',gray_input=True)
    print(input_array.shape)
    
    out=onnxrun(onnx_path,input_array)[0] #onnx-graph has only 1 output
    print('out.shape ',out.shape)
    print("out ",out[0][:3])


    img_path='./img_OUT_0_resize.ppm'
    yolo_proc(img_path,onnx_path,conf_thres=0.2,iou_thres=0.45)
    


    