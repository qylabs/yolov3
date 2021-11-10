import numpy as np
import onnxruntime as ort
import cv2
import copy
np.set_printoptions(threshold=np.inf)

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


def main(img_path,onnx_path,gray_input):
    '''
    single one img
    '''
    img0=cv2.imread(img_path)
    print(img0.shape)
    #preproc img
    input_array=transfrom_img(img_path,gray_input)
    print(input_array.shape)
    print(input_array)
    out=onnxrun(onnx_path,input_array)[0] #onnx-graph has only 1 output
    print('==>>out.shape ',out.shape)
    # print('===out:\n',out)
    return out


if __name__=="__main__":
    onnx_path='./checkpoint/mobilenet_cust_v1_1_market1501_pretrained.onnx'

    img_path='./test_sample/wk.ppm'

    # compute ONNX Runtime output prediction
    batch_size=1
    channels=3#1
    height=128
    width=64

    out=main(img_path,onnx_path,gray_input=False)
    np.savetxt(img_path.replace('.ppm','.txt'),out.flatten())
