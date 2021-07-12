import numpy as np
import onnxruntime as ort
import cv2


def transfrom_img(img_path):
    img=cv2.imread(img_path)
    mean=np.mean(img,axis=(0,1))
    std=np.std(img,axis=(0,1))
    final=np.transpose((img-mean)/std,(2,0,1))
    final=np.expand_dims(final,axis=0)
    return final.astype(np.float32)


def onnxrun(onnx_path,input_array):
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_array}
    ort_outs = ort_session.run(None, ort_inputs)
    print('len(ort_outs) ',len(ort_outs))
    return ort_outs


if __name__=="__main__":
    # onnx_path='./weights/yolov3-tiny_sim.onnx'
    # onnx_path='./weights/yolov3-tiny_sim_relu.onnx'
    onnx_path='./weights/onnx_model_zoo/tiny-yolov3-11.onnx'

    # compute ONNX Runtime output prediction
    batch_size=1
    channels=3
    height=320
    width=320

    input_array=np.random.randn(batch_size, channels, height, width).astype(np.float32)
    
    out=onnxrun(onnx_path,input_array)

