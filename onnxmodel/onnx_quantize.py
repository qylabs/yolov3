import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
 
model_fp32 = './runs/train/finetune_coconoperson/weights/best.onnx'
model_quant = './runs/train/finetune_coconoperson/weights/int8.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)