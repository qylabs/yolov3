import onnx
import onnxsim


# model_onnx='./weights/onnx_model_zoo/tiny-yolov3-11.onnx'
model_onnx='./weights/onnx_model_zoo/yolov3-10.onnx'
dynamic=True
if dynamic:
    name='input_1'
    # img_shape=[1,3,320,320]
    img_shape=[1,3,640,640]

print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
model_onnx, check = onnxsim.simplify(model_onnx,
                                    dynamic_input_shape=dynamic,
                                    input_shapes={name: img_shape} if dynamic else None)
assert check, 'assert check failed'

f=model_onnx.replace('.onnx','_sim.onnx')
onnx.save(model_onnx, f)